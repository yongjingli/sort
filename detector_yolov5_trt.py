import tensorrt as trt   # tensorRT 7.0
import pycuda.driver as cuda
import pycuda.autoinit

from datetime import datetime
import numpy as np
import cv2
import time
import os

# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # set batch size

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Yolov5DetectorTrt():
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.engine = self.build_engine_onnx(onnx_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

        self.img_scale = 1.0
        self.pad_h = 0
        self.pad_w = 0

        self.conf = 0.7
        print('Yolov5DetectorTrt Init Done.')

    def GiB(self, val):
        return val * 1 << 30

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def build_engine_onnx(self, model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_workspace_size = self.GiB(1)
            builder.max_batch_size = 1
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            return builder.build_cuda_engine(network)

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            bindig_shape = tuple(engine.get_binding_shape(binding))
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(bindig_shape, dtype)
            # print('\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))  # host mem

            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # print('\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem))) # device mem

            # print('\t# Append the device buffer to device bindings.......')
            bindings.append(int(device_mem))
            # print('\tbindings: ', bindings)

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                # print("this is the input!")
                # print('____HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                # print("This is the output!")
                outputs.append(HostDeviceMem(host_mem, device_mem))
            # print("----------------------end allocating one binding in the onnx model-------------------------")

        return inputs, outputs, bindings, stream

    def onnx_input_process(self, img_src, focus=False):
        fix_h, fix_w = 640, 640
        img_temp = img_src[..., ::-1]  # BGR to RGB
        scale = float(fix_h) / float(max(img_src.shape[0], img_src.shape[1]))

        img_temp = cv2.resize(img_temp, (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale)),
                              interpolation=cv2.INTER_NEAREST)
        img_src = cv2.resize(img_src, (int(img_src.shape[1] * scale), int(img_src.shape[0] * scale)),
                             interpolation=cv2.INTER_NEAREST)

        # cal pad_w, and pad_h
        pad_h = (fix_h - img_temp.shape[0]) // 2
        pad_w = (fix_w - img_temp.shape[1]) // 2

        self.pad_h = pad_h
        self.pad_w = pad_w
        self.img_scale = scale

        # padding
        img_input = np.ones((fix_h, fix_w, 3), dtype=np.float32) * 0
        img_input[pad_h:img_temp.shape[0] + pad_h, pad_w:img_temp.shape[1] + pad_w, :] = img_temp

        # Convert
        img_input /= 255.0
        img_input = img_input.transpose(2, 0, 1)  # to C, H, W
        img_input = np.ascontiguousarray(img_input)
        #
        img_input = np.expand_dims(img_input, 0)

        # foucs op
        if focus:
            img_input = np.concatenate(
                (img_input[..., ::2, ::2], img_input[..., 1::2, ::2], img_input[..., ::2, 1::2],
                 img_input[..., 1::2, 1::2]), 1)

        img_input.astype(np.float32)
        return img_input

    def trt_cv_img_preprocess(self, img_src, pagelocked_buffer):
        # Converts the input image to a CHW Numpy array
        trt_inputs = self.onnx_input_process(img_src, focus=True)
        np.copyto(pagelocked_buffer, trt_inputs)
        return img_src

    def parse_onnx_feats(self, feats):
        stride = [8., 16., 32.]
        anchor = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                          dtype=np.float32)
        anchor_grid = anchor.reshape(len(stride), 1, -1, 1, 1, 2)

        max_det = 300
        z = []
        for i, feat in enumerate(feats):
            # print(output)
            bs, na, ny, nx, no = feat.shape
            yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
            grid = grid[..., ::-1]

            # bs, na, ny, nx, no = feature.shape
            # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(device)

            y = self.sigmoid(feat)

            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy

            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, no))

        prediction = np.concatenate(z, axis=1)
        prediction = prediction.astype(np.float32)
        return prediction

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction):
        # min_conf = 0.4
        cover_iou = 0.5
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > self.conf  # candidates

        output = []
        # batch size = 1
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence filter

            # If none remain process next image
            if not x.shape[0]:
                continue

            # # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # if xi == 0:
            #     print(x[:, 5])

            # # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # # Detections matrix nx6 (xyxy, conf, cls)
            j = np.argmax(x[:, 5:], axis=1)
            conf = np.max(x[:, 5:], axis=1).astype(np.float32)

            j = np.expand_dims(j, axis=1)
            conf = np.expand_dims(conf, axis=1)

            x = np.concatenate([box, conf, j], axis=1)[conf.reshape(-1) > self.conf]
            conf_sort = np.argsort(-x[:, 4], axis=0)  # sorted by confident
            x = x[conf_sort, :]
            skip = [False] * len(conf_sort)
            mask = [False] * len(conf_sort)
            for i in range(len(conf_sort)):
                if skip[i]:
                    continue
                mask[i] = True
                box_high = x[i]
                for j in range(i + 1, len(conf_sort)):
                    box_comp = x[j]
                    if skip[j] or box_high[5] != box_comp[5]:
                        continue
                    xx1 = max(box_high[0], box_comp[0])
                    yy1 = max(box_high[1], box_comp[1])
                    xx2 = min(box_high[2], box_comp[2])
                    yy2 = min(box_high[3], box_comp[3])
                    iou_w = xx2 - xx1 + 1
                    iou_h = yy2 - yy1 + 1
                    if iou_h > 0 and iou_w > 0:
                        area_high = (box_high[2] - box_high[0]) * (box_high[3] - box_high[1])
                        area_comp = (box_comp[2] - box_comp[0]) * (box_comp[3] - box_comp[1])
                        over = iou_h * iou_w / (area_high + area_comp - iou_h * iou_w)
                        if over > cover_iou:
                            skip[j] = True
            x = x[mask, :]
            output = x
        return output

    def scale_pred_det(self, pred_det):
        for pred_ in pred_det:
            pred_[0] = int((pred_[0] - self.pad_w) / self.img_scale)
            pred_[2] = int((pred_[2] - self.pad_w) / self.img_scale)
            pred_[1] = int((pred_[1] - self.pad_h) / self.img_scale)
            pred_[3] = int((pred_[3] - self.pad_h) / self.img_scale)
        return pred_det

    def infer_cv_img(self, cv_img):
        test_image = self.trt_cv_img_preprocess(cv_img, self.inputs[0].host)
        # start_time = time.time()
        trt_outputs = self.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        prediction = self.parse_onnx_feats(trt_outputs)
        prediction = self.non_max_suppression(prediction)
        pred_det = self.scale_pred_det(prediction)

        # end_time = time.time() - start_time
        # print('do inference time:{}'.format(end_time))
        self.pred_det_scale = pred_det
        return pred_det


def detect_imgs(onnx_path, img_dir):
    yolo_detector_trt = Yolov5DetectorTrt(onnx_path)
    save_img_dir = os.path.join(img_dir, 'det_result')
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    img_names = list(filter(lambda x: os.path.splitext(x)[-1], os.listdir(img_dir)))
    for img_name in img_names:
        print('proc {}:'.format(img_name))
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        pred_det = yolo_detector_trt.infer_cv_img(img)
        for pred_ in pred_det:
            if pred_[5] != 0:
                continue

            show_color = (0, 0, 255)
            cv2.putText(img, str(round(pred_[4], 2)), (int(pred_[0]), int(pred_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, show_color, 1)
            cv2.rectangle(img, (int(pred_[0]), int(pred_[1])), (int(pred_[2]), int(pred_[3])), show_color, 2)

        save_img_path = os.path.join(save_img_dir, img_name)
        cv2.imwrite(save_img_path, img)


if __name__ == '__main__':
    print('Start Proc.')
    onnx_path = '/home/liyongjing/Egolee/programs/yolov5-master/weights/best_sim.onnx'
    img_dir = '/home/liyongjing/Egolee/data/test_person_car/wubao_all_collect/tmp_alarm'
    detect_imgs(onnx_path, img_dir)


    # yolo_detector_trt = Yolov5DetectorTrt(onnx_path)
    #
    # i = 0
    # while(i < 10):
    #     img = cv2.imread(img_path)
    #
    #     start_time = time.time()
    #     pred_det = yolo_detector_trt.infer_cv_img(img)
    #     end_time = time.time() - start_time
    #     print('infer time:{}s'.format(round(end_time, 4)))
    #
    #     # visualize
    #     for pred_ in pred_det:
    #         show_color = (255, 0, 255)
    #        proc if pred_[5] == 0:
    #             show_color = (0, 255, 0)
    #         cv2.putText(img, str(round(pred_[4], 2)), (int(pred_[0]), int(pred_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, show_color, 1)
    #         cv2.rectangle(img, (int(pred_[0]), int(pred_[1])), (int(pred_[2]), int(pred_[3])), show_color, 2)
    #     cv2.namedWindow("img", 0)
    #     cv2.imshow("img", img)
    #     wait_key = cv2.waitKey(0)
    #     i = i + 1

    print('End Proc.')  
