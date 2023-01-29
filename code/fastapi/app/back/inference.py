from mmdet.apis import inference_detector
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
from copy import deepcopy


class Inference:
    def __init__(self, images, exam_info, coco, detector):
        self.images = images
        self.coco = coco
        self.detector = detector
        self.exam_info = self.load_exam_info(exam_info)
        self.inference_detector = inference_detector

    def load_exam_info(self, exam_info):
        """
        img info를 불러오는 함수

        Args:
            exam_info (str): 시험지의 정보 {년도}_{몇월}_{형}
                            ex) 2013_9_a or 2022_f

            coco : 사전에 제작된 annotation기반 coco format 데이터

        Returns:
            List: exam_info에 대응하는 img파일들의 사전 정보
            ex) : exam_info = 2012_9_a 일때 12년 9월 a형에 문제 정보가 p1 ~ 마지막 페이지까지 리스트에 담김
                [2012_9_a_p1 정보, 2012_9_a_p2 정보 .....]
        """
        all_img_info = self.coco.loadImgs(self.coco.getImgIds())
        result = [info for info in all_img_info if exam_info in info["file_name"]]
        result = sorted(result, key=lambda x: x["file_name"])
        return result

    def load_anns(self, idx):
        img_info = self.exam_info[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        return img_info, anns

    def load_anns_q(self, idx):
        """
        시험의 정보와 이미지를 받아 그것에 대응되는 사전에 annotation된 questions을 반환
        """
        _, anns = self.load_anns(idx)
        questions = [ann for ann in anns if ann["category_id"] > 6]
        anns_dict = {q["category_id"]: self.xywh2ltrb(q["bbox"]) for q in questions}
        return anns_dict

    def get_predict(self, img, box_threshold=0.3):
        """_summary_

        Args:
            img: img 파일의 경로
            detector: 사전에 제작된 mmdetection 모델
            box_threshold (float, optional): detector로 예측된 box들의 최소 임계값

        Returns:
            List: [np.array(left,top,right,bottom,label) .... ]
            list안의 값은 np.array 형식으로 박스정보와 label 값이 int type으로 정의됨
        """
        inference = self.inference_detector(self.detector, img)
        predict = []
        for label, bboxes in enumerate(inference):
            predict += [
                np.append(bbox[:4], [bbox[4], label])
                for bbox in bboxes
                if (bbox[4] > box_threshold)
            ]
        predict = sorted(predict, key=lambda x: x[4], reverse=True)
        return predict

    def xywh2ltrb(self, bbox):
        """
        Args: (x,y)는 좌측 상단
            bbox: (x,y,w,h)

        Returns:
            bbox: (left,top,right,bottom)
        """
        return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

    def compute_iou(self, box1, box2):
        """iou 계산

        Args:
            box :(List): [left,top,right,bottom]

        Returns:
            iou(float): box1과 box2의 iou값 반환
        """
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box1_area + box2_area - intersection

        iou = intersection / union
        assert iou <= 1, "iou값이 1 초과"
        return iou

    def match_qa(self, anns):
        """
        사전에 정의된 annotation을 기반으로 같은 문제의 question 박스와 answer 박스를 매칭
        Args:
            anns : 사전에 정의된 annotation 정보

        Returns:
            anns_dict (dict): 문제에 대응하는 정답 박스의 위치 반환
            key : category_id
            value : answer box의 정보 : left,top,right,bottom
        """

        questions = [ann for ann in anns if ann["category_id"] > 6]
        answers = [ann for ann in anns if ann["category_id"] <= 6]

        # 추후에 중복되는 연산 제거하는 코드로 수정하기!!
        anns_dict = {}
        for q in questions:
            q_ = self.xywh2ltrb(q["bbox"])
            for a in answers:
                a_ = self.xywh2ltrb(a["bbox"])
                if self.compute_iou(q_, a_) > 0:
                    anns_dict[q["category_id"]] = a_
                    break

        return anns_dict

    def make_qa(self, predict, anns):
        """문제에 대응하는 최적의 answer box 찾기
           최적의 answer box 기준은 사전의 annotation결과와 iou가 가장 높은 box
           iou 값이 0.3 이상인 박스가 없는경우 예측을 못했다고 판단하고 결과로 (0,0,0,0,0,0)을 반환

        Args:
            predict (List): [[box_info, confidence score,label] ....] box_info = left,top,right,bottom
            anns (dict): key : 문제 categori id, value: bbox left,top,right,bottom

        Returns:
            question_answer (dict):  key : 문제번호, value : [boxinfo, categori_id]
        """
        question_answer = {}
        for q, bbox in anns.items():
            iou_list = [self.compute_iou(pred, bbox) for pred in predict]
            max_iou = max(iou_list) if iou_list else 0
            if max_iou < 0.3:
                question_answer[q - 6] = np.array([0, 0, 0, 0, 0, -1])
            else:
                question_answer[q - 6] = predict[iou_list.index(max_iou)]
        return question_answer

    def resize_box(self, q_a_box, img_info, img_shape):
        """input의 이미지 size와 사전에 정의된 annotation의 box사이즈를 대응시킴

        Args:
            q_a_box : dict : key = 문제, value : 문제에 대응하는 정답박스 정보
            img_info : 사전에 정의된 annotation 이미지 정보
            img_shape : input 이미지의 사이즈

        Returns:
            annotation의 size를 적용한
            q_a_box : dict : key = 문제, value : 문제에 대응하는 정답박스 정보
        """
        w_r = img_shape[1] / img_info["width"]
        h_r = img_shape[0] / img_info["height"]

        q_a_box = {
            q: (int(a[0] * w_r), int(a[1] * h_r), int(a[2] * w_r), int(a[3] * h_r))
            for q, a in q_a_box.items()
        }
        return q_a_box

    def save_predict(self, img, img_path, qa_info):
        """
        predicted img 저장하는 함수!
        """
        for bbox in qa_info.values():
            left, top, right, bottom = bbox[:4].astype(int)
            cv2.putText(
                img,
                f"{int(bbox[-1])}   {bbox[-2]:.4f}",
                (left, top - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (255, 0, 0),
                3,
            )
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
            cv2.imwrite(
                f"/opt/ml/input/code/fastapi/app/log/{img_path}_predict.jpg", img
            )

    def make_user_solution(self, img_save=False, log_save=False):
        """
        Args:
            img_save (bool): 예측한 결과의 사진을 저장할지 여부
            log_save(bool): 이미지의 정답과 예측값 그 값에 대응하는 confidence csv 저장

        Returns:
            dict: key 문제번호, value : 예측한 체크박스의 번호
        """
        answer_bbox, a_label = {}, []
        images = deepcopy(self.images)
        for idx, img in enumerate(images):
            img_info, anns = self.load_anns(idx)
            predict = self.get_predict(img)
            q_a_box = self.match_qa(anns)
            q_a_box = self.resize_box(q_a_box, img_info, img.shape)
            qa_info = self.make_qa(predict, q_a_box)
            answer_bbox.update(qa_info)

            if img_save:
                self.save_predict(img, idx, qa_info)

            if log_save:
                a_label += [ann for ann in anns if ann["category_id"] <= 6]

        if log_save:
            a_b = sorted(answer_bbox.items())
            pred_data = pd.DataFrame(
                {
                    "label": [a["category_id"] - 1 for a in a_label],
                    "predict": [int(bbox[-1]) for q, bbox in a_b],
                    "confidence": [bbox[-2] for q, bbox in a_b],
                }
            )
            if not os.path.isdir("/opt/ml/input/code/fastapi/app/log"):
                os.mkdir("/opt/ml/input/code/fastapi/app/log")
            pred_data.to_csv("/opt/ml/input/code/fastapi/app/log/predict_log.csv")

        user_solution = {q: int(bbox[-1]) for q, bbox in sorted(answer_bbox.items())}
        return user_solution

    def save_score_img(self, scoring_result):
        # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
        # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
        o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/correct.png")
        x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/wrong.png")
        o_width, o_height = o_image.size
        x_width, x_height = x_image.size

        # TODO: 현재 paste 좌표가 좌측 하단으로 잡혀있음 (좌측 상단으로 바꿔야함. annotation 정보 확인 필요)
        score_img = []
        for idx, img in enumerate(self.images):  # fix
            background = Image.fromarray(img)
            question_ann = self.load_anns_q(idx)
            for cat_id, bbox in question_ann.items():
                question = cat_id - 6  # 문제 번호: 1 ~ 30
                if scoring_result[question] == "O":
                    background.paste(
                        o_image,
                        (
                            int(bbox[0] - o_width / 2),
                            int(bbox[1] - o_height / 2) + 10,
                        ),
                        o_image,
                    )
                elif scoring_result[question] == "X":
                    background.paste(
                        x_image,
                        (
                            int(bbox[0] - x_width / 2),
                            int(bbox[1] - x_height / 2) + 10,
                        ),
                        x_image,
                    )
            score_img.append(background)
        return score_img


if __name__ == "__main__":
    from pycocotools.coco import COCO
    from mmdet.apis import init_detector
    import os
    from PIL import Image

    model_config = "/opt/ml/input/data/models/19/config.py"
    model_weight = "/opt/ml/input/data/models/19/model.pth"
    coco = COCO("/opt/ml/input/data/annotations/train_v1-3.json")
    detector = init_detector(model_config, model_weight, device="cuda:0")

    img_path = "/opt/ml/input/data/infer_test"
    img_paths = [os.path.join(img_path, img) for img in os.listdir(img_path)]
    images_np = [np.array(Image.open(img)) for img in img_paths]
    inference = Inference(
        images=images_np,
        exam_info="2021_f_a",
        coco=coco,
        detector=detector,
    )
    result = inference.make_user_solution(True, True)
    print("hi")


class Inference_v2:
    def __init__(self, images, detector):
        self.images = images
        self.detector = detector
        self.inference_detector = inference_detector

    def get_predict(self, img):
        """_summary_

        Args:
            img: numpy ndarray 형식의 이미지
            detector: 사전에 제작된 mmdetection 모델

        Returns:
            List: [[left,top,right,bottom,label], .... ]
            list안의 값은 list 형식으로 박스정보와 label 값이 int type으로 정의됨
        """
        inference = self.inference_detector(self.detector, img)
        predict = []
        for label, bboxes in enumerate(inference):
            predict += [
                [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], label]
                for bbox in bboxes
            ]
        predict = sorted(predict, key=lambda x: x[4], reverse=True)
        return predict

    def compute_iou(self, box1, box2):
        """iou 계산

        Args:
            box(List): [left,top,right,bottom]

        Returns:
            iou(float): box1과 box2의 iou값 반환
        """
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box1_area + box2_area - intersection

        iou = intersection / union
        assert iou <= 1, "iou값이 1 초과"
        return iou

    def save_predict(self, img, idx, predict, num):
        """predicted img 저장

        Args:
            img(numpy.ndarray): 형식의 이미지
            idx(int): 저장될 파일의 index
            predict(list): 예측 결과
            num(int): 문제 번호
        """
        for bboxes in predict:
            for bbox in bboxes:
                left, top, right, bottom, confidence_score, label = bbox[:6]
                if label == 6:
                    label = f'q{num}'
                    num += 1
                cv2.putText(
                    img,
                    f"{label}   {confidence_score:.4f}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (255, 0, 0),
                    3,
                )
                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
            cv2.imwrite(
                f"/opt/ml/input/code/fastapi/app/log/{idx}_predict.jpg", img
            )
    
    def match(self, predict):
        """문제와 보기를 연결하는 함수

        Args:
            predict(list): 문제와 보기가 섞인 예측 결과
        
        Returns:
            qa_list(list): [[[1번 문제], [1번 보기]], [[2번 문제], [2번 보기]], ...]

        """
        # 문제와 보기를 나누고 각각 겹치는 걸 계산
        qa_list = []
        # 좌, 우 / 문제, 보기 나누기
        left_side_q, right_side_q = [], []
        left_side_a, right_side_a = [], []
        for p in predict:
            if p[0] < self.images[0].shape[1] / 2:
                if p[5] == 6:
                    left_side_q.append(p)
                else:
                    left_side_a.append(p)
            else:
                if p[5] == 6:
                    right_side_q.append(p)
                else:
                    right_side_a.append(p)
        left_side_q, right_side_q = sorted(left_side_q, key=lambda x: -x[4]), sorted(right_side_q, key=lambda x: -x[4])
        left_side_a, right_side_a = sorted(left_side_a, key=lambda x: -x[4]), sorted(right_side_a, key=lambda x: -x[4])

        left_side = [[left_side_q[0]]]
        for i in range(1, len(left_side_q)):
            if self.compute_iou(left_side_q[0], left_side_q[i]) < 0.1:
                left_side.append([left_side_q[i]])
                break
        if left_side_a:
            for i in range(len(left_side)):
                tmp = []
                for j in range(len(left_side_a)):
                    if self.compute_iou(left_side[i][0], left_side_a[j]) > 0:
                        tmp.append(left_side_a[j])
                tmp = sorted(tmp, key=lambda x: -x[4])
                left_side[i].append(tmp[0])
        qa_list += sorted(left_side, key=lambda x: x[0][1])
        
        right_side = [[right_side_q[0]]]
        for i in range(1, len(right_side_q)):
            if self.compute_iou(right_side_q[0], right_side_q[i]) < 0.1:
                right_side.append([right_side_q[i]])
                break
        if right_side_a:
            for i in range(len(right_side)):
                tmp = []
                for j in range(len(right_side_a)):
                    if self.compute_iou(right_side[i][0], right_side_a[j]) > 0:
                        tmp.append(right_side_a[j])
                tmp = sorted(tmp, key=lambda x: -x[4])
                right_side[i].append(tmp[0])
        qa_list += sorted(right_side, key=lambda x: x[0][1])
        return qa_list

    def make_user_solution(self, img_save=False, log_save=False):
        """
        Args:
            img_save(bool): 예측한 결과의 사진을 저장할지 여부
            log_save(bool): 이미지의 보기, 정답의 예측값과 그 값에 대응하는 confidence를 csv로 저장

        Returns:
            user_solution(dict): key=문제번호, value=예측한 답
            q_bbox(lst): 문제 박스의 bbox 좌표가 담긴 리스트(최종 output 이미지를 위한)
        """
        a_label, q_label, q_bbox = [], [], []
        images = deepcopy(self.images)

        num = 1
        for idx, img in enumerate(images):
            predict = self.get_predict(img)
            predict = self.match(predict)
            for bbox in predict:
                for b in bbox:
                    for i in range(4):
                        b[i] = int(b[i])
            if img_save:
                self.save_predict(img, idx, predict, num)
            q_bbox.append([])
            for ps in predict:
                if len(ps) == 1:
                    q_label.append(ps[0])
                    a_label.append([0,0,0,0,0,-1])
                    q_bbox[idx].append(ps[0][:4])
                else:
                    for p in ps:
                        if p[-1] == 6:
                            q_label.append(p)
                            q_bbox[idx].append(p[:4])
                        else:
                            a_label.append(p)
            num += len(predict)
        if log_save:
            pred_data = pd.DataFrame([ x for x in zip([i for i in range(1, len(q_label)+1)], [x[-2] for x in q_label], [x[-1] for x in a_label], [x[-2] for x in a_label])], columns=['question', 'confidence_q', 'answer', 'confidence_a'])
            if not os.path.isdir("/opt/ml/input/code/fastapi/app/log"):
                os.mkdir("/opt/ml/input/code/fastapi/app/log")
            pred_data.to_csv("/opt/ml/input/code/fastapi/app/log/predict_log.csv", index=False)

        user_solution = {i+1: a[-1] for i, a in enumerate(a_label)}
        return user_solution, q_bbox

    def save_score_img(self, scoring_result, q_bbox):
        # 채점된 이미지를 만들기 위해 o, x 이미지를 불러오는 부분입니다.
        # TODO: 위의 input이미지의 resize 부분과 함께 고려해야 할 사항입니다.
        o_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/correct.png")
        x_image = Image.open("/opt/ml/input/code/fastapi/app/scoring_image/wrong.png")
        o_width, o_height = o_image.size
        x_width, x_height = x_image.size

        # TODO: 현재 paste 좌표가 좌측 하단으로 잡혀있음 (좌측 상단으로 바꿔야함. annotation 정보 확인 필요)
        score_img = []
        num_q = 1
        for i, img in enumerate(self.images):  # fix
            background = Image.fromarray(img)
            for bbox in q_bbox[i]:
                if scoring_result[num_q] == "O":
                    background.paste(
                        o_image,
                        (
                            int(bbox[0] - o_width / 2),
                            int(bbox[1] - o_height / 2) + 10,
                        ),
                        o_image,
                    )
                elif scoring_result[num_q] == "X":
                    background.paste(
                        x_image,
                        (
                            int(bbox[0] - x_width / 2),
                            int(bbox[1] - x_height / 2) + 10,
                        ),
                        x_image,
                    )
                num_q += 1
            score_img.append(background)
        return score_img