import json
import pandas as pd
import psycopg2


# db 연결
db_secrets_path = "/opt/ml/input/code/fastapi/app/back/db_secrets.json"
with open(db_secrets_path, "r") as f:
    db_secrets = json.load(f)


def get_info_from_db(exam_info):
    db = psycopg2.connect(
        host=db_secrets["DB"]["host"],
        dbname=db_secrets["DB"]["dbname"],
        user=db_secrets["DB"]["user"],
        password=db_secrets["DB"]["password"],
        port=db_secrets["DB"]["port"],
    )

    answer = pd.read_sql(
        f"""
        select question, answer 
        from answer_sheet 
        where type 
        like \'%{exam_info}%\';
        """,
        db,
    )

    question = pd.read_sql(
        f"""
        select question, x_question, y_question, w_question, h_question, page 
        from question_bbox 
        where type 
        like '%{exam_info}%';
        """,
        db,
    )

    img_shape = pd.read_sql(
        f"""
        select width, height 
        from page_shape 
        where type 
        like \'%{exam_info}%\';
        """,
        db,
    )

    answer = {q: a for q, a in zip(answer["question"], answer["answer"])}
    q_bbox = [[dict()] for _ in range(max(question["page"]))]
    for q, x, y, w, h, page in zip(
        question["question"],
        question["x_question"],
        question["y_question"],
        question["w_question"],
        question["h_question"],
        question["page"],
    ):
        q_bbox[page - 1][0].update({q: [x, y, w, h]})
    img_shape = [img_shape["width"][0], img_shape["height"][0]]

    db.close()
    return answer, q_bbox, img_shape


def insert_log(log_pred, exam_info, time):
    db = psycopg2.connect(
        host=db_secrets["DB"]["host"],
        dbname=db_secrets["DB"]["dbname"],
        user=db_secrets["DB"]["user"],
        password=db_secrets["DB"]["password"],
        port=db_secrets["DB"]["port"],
    )

    for p in log_pred:
        x_pred, y_pred, w_pred, h_pred, confidence, pred, question = p
        db.cursor().execute(
            f"""
            insert into log (type, question, infer_date, x_pred, y_pred, w_pred, h_pred, confidence, class_pred) 
            values (\'{exam_info}\', {question}, \'{time}\', {x_pred}, {y_pred}, {w_pred}, {h_pred}, {confidence}, {pred});
            """
        )
        db.commit()
    db.close()
