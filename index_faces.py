# common dependencies
import os
from os import path
import warnings
import time
import pickle
import logging
from deepface import DeepFace
import uuid

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf

# import faiss
# from prisma import Prisma
import time

# package dependencies
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    SFace,
)
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)


def find(
    img_path,
    db_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):
    """
    This function applies verification several times and find the identities in a database

    Parameters:
            img_path: exact image path, numpy array (BGR) or based64 encoded image.
            Source image can have many faces. Then, result will be the size of number of
            faces in the source image.

            db_path (string): You should store some image files in a folder and pass the
            exact folder path to this. A database image can also have many faces.
            Then, all detected faces in db side will be considered in the decision.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
            Dlib, ArcFace, SFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (bool): The function throws exception if a face could not be detected.
            Set this to True if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

            silent (boolean): disable some logging and progress bars

    Returns:
            This function returns list of pandas data frame. Each item of the list corresponding to
            an identity in the img_path.
    """

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path + "/" + file_name):
        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print(
                "There are ",
                len(representations),
                " representations found in ",
                file_name,
            )

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
            disable=silent,
        )
        for index in pbar:
            employee = employees[index]

            img_objs = functions.extract_faces(
                img=employee,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            for img_content, _, _ in img_objs:
                embedding_obj = DeepFace.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                # print(instance)
                representations.append(instance)
                # break
        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new identities in your database."
            )

    # ----------------------------
    # now, we got representations for facial database

    df = pd.DataFrame(
        representations, columns=["identity", f"{model_name}_representation"]
    )

    # img path might have more than once face
    target_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    resp_obj = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = DeepFace.represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= 0.285]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return resp_obj


async def index_faces(
    db_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):
    """
    This function applies verification several times and find the identities in a database

    Parameters:
            img_path: exact image path, numpy array (BGR) or based64 encoded image.
            Source image can have many faces. Then, result will be the size of number of
            faces in the source image.

            db_path (string): You should store some image files in a folder and pass the
            exact folder path to this. A database image can also have many faces.
            Then, all detected faces in db side will be considered in the decision.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
            Dlib, ArcFace, SFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (bool): The function throws exception if a face could not be detected.
            Set this to False if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

            silent (boolean): disable some logging and progress bars

    Returns:
            This function returns list of pandas data frame. Each item of the list corresponding to
            an identity in the img_path.
    """

    tic = time.time()
    db = Prisma()
    await db.connect()
    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    # file_name = f"representations_{model_name}.pkl"
    # file_name = file_name.replace("-", "_").lower()

    # if path.exists(db_path + "/" + file_name):
    #     if not silent:
    #         print(
    #             f"WARNING: Representations for images in {db_path} folder were previously stored"
    #             + f" in {file_name}. If you added new instances after the creation, then please "
    #             + "delete this file and call find function again. It will create it again."
    #         )

    #     with open(f"{db_path}/{file_name}", "rb") as f:
    #         representations = pickle.load(f)

    #     if not silent:
    #         print(
    #             "There are ",
    #             len(representations),
    #             " representations found in ",
    #             file_name,
    #         )

    # else:  # create representation.pkl from scratch
    employees = []

    for r, _, f in os.walk(db_path):
        for file in f:
            if (
                (".jpg" in file.lower())
                or (".jpeg" in file.lower())
                or (".png" in file.lower())
            ):
                exact_path = r + "/" + file
                employees.append(exact_path)

    if len(employees) == 0:
        raise ValueError(
            "There is no image in ",
            db_path,
            " folder! Validate .jpg or .png files exist in this path.",
        )

    # ------------------------
    # find representations for db images

    representations = []

    # for employee in employees:
    pbar = tqdm(
        range(0, len(employees)),
        desc="Finding representations",
        disable=silent,
    )
    for index in pbar:
        employee = employees[index]

        img_objs = functions.extract_faces(
            img=employee,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )

        for img_content, _, _ in img_objs:
            embedding_obj = DeepFace.represent(
                img_path=img_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img_representation = embedding_obj[0]["embedding"]
            face = await db.face.create(
                {"embedding": img_representation, "image_url": employee}
            )

            representations.append(face)

            # instance = []
            # instance = {
            #     "id": uuid.uuid4(),
            #     "embedding": img_representation,
            #     "image": employee,
            # }

            # instance.append(employee)
            # representations.append(instance)
            # representations.append(instance)

    # -------------------------------

    # with open(f"{db_path}/{file_name}", "wb") as f:
    #     pickle.dump(representations, f)

    # if not silent:
    #     print(
    #         f"Representations stored in {db_path}/{file_name} file."
    #         + "Please delete this file when you add new identities in your database."
    #     )

    # ----------------------------
    # now, we got representations for facial database
    # df = pd.DataFrame(
    #     representations, columns=["identity", f"{model_name}_representation"]
    # )

    # # img path might have more than once face
    # target_objs = functions.extract_faces(
    #     img=img_path,
    #     target_size=target_size,
    #     detector_backend=detector_backend,
    #     grayscale=False,
    #     enforce_detection=enforce_detection,
    #     align=align,
    # )

    # resp_obj = []

    # for target_img, target_region, _ in target_objs:
    #     target_embedding_obj = DeepFace.represent(
    #         img_path=target_img,
    #         model_name=model_name,
    #         enforce_detection=enforce_detection,
    #         detector_backend="skip",
    #         align=align,
    #         normalization=normalization,
    #     )

    #     target_representation = target_embedding_obj[0]["embedding"]

    #     result_df = df.copy()  # df will be filtered in each img
    #     result_df["source_x"] = target_region["x"]
    #     result_df["source_y"] = target_region["y"]
    #     result_df["source_w"] = target_region["w"]
    #     result_df["source_h"] = target_region["h"]

    #     distances = []
    #     for index, instance in df.iterrows():
    #         source_representation = instance[f"{model_name}_representation"]

    #         if distance_metric == "cosine":
    #             distance = dst.findCosineDistance(
    #                 source_representation, target_representation
    #             )
    #         elif distance_metric == "euclidean":
    #             distance = dst.findEuclideanDistance(
    #                 source_representation, target_representation
    #             )
    #         elif distance_metric == "euclidean_l2":
    #             distance = dst.findEuclideanDistance(
    #                 dst.l2_normalize(source_representation),
    #                 dst.l2_normalize(target_representation),
    #             )
    #         else:
    #             raise ValueError(f"invalid distance metric passes - {distance_metric}")

    #         distances.append(distance)

    #         # ---------------------------

    #     result_df[f"{model_name}_{distance_metric}"] = distances

    #     threshold = dst.findThreshold(model_name, distance_metric)
    #     result_df = result_df.drop(columns=[f"{model_name}_representation"])
    #     result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
    #     result_df = result_df.sort_values(
    #         by=[f"{model_name}_{distance_metric}"], ascending=True
    #     ).reset_index(drop=True)

    #     resp_obj.append(result_df)
    # # -----------------------------------

    # toc = time.time()

    # if not silent:
    #     print("find function lasts ", toc - tic, " seconds")

    # return {
    #     "is_trained": index.is_trained,
    #     "ntotal": index.ntotal,
    #     "data": representations,
    #     "index": index,
    # }

    embeddings = [face.embedding for face in representations]

    try:
        index = load_faiss_index("faces.index")
        embeddings_np = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_np)
        save_faiss_index(index, "faces.index")
    except Exception as e:
        await build_faiss_index(embeddings)

    return representations


async def index_faces(
    img_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):
    # query_embed = DeepFace.represent(
    #     img_path,
    #     detector_backend="retinaface",
    #     # model_name="Facenet",
    # )
    # query_embedding = query_embed[0]["embedding"]

    # print(query_embedding)

    db = Prisma()
    await db.connect()

    # employees = []

    # for r, _, f in os.walk(db_path):
    #     for file in f:
    #         if (
    #             (".jpg" in file.lower())
    #             or (".jpeg" in file.lower())
    #             or (".png" in file.lower())
    #         ):
    #             exact_path = r + "/" + file
    #             employees.append(exact_path)

    # if len(employees) == 0:
    #     raise ValueError(
    #         "There is no image in ",
    #         db_path,
    #         " folder! Validate .jpg or .png files exist in this path.",
    #     )

    # representations = []

    # # for employee in employees:
    # pbar = tqdm(
    #     range(0, len(employees)),
    #     desc="Finding representations",
    #     disable=silent,
    # )
    # for index in pbar:
    #     employee = employees[index]

    representaions = []

    img_objs = functions.extract_faces(
        img=img_path,
        # target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img_content, _, _ in img_objs:
        embedding_obj = DeepFace.represent(
            img_path=img_content,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        img_representation = embedding_obj[0]["embedding"]
        face = await db.face.create(
            {"embedding": img_representation, "image_url": img_path}
        )

        representaions.append(face)

    return representaions

    #         print(employee)

    #         # instance = []
    #         instance = {
    #             "id": uuid.uuid4(),
    #             "embedding": img_representation,
    #             "image": employee,
    #         }

    #         # instance.append(employee)
    #         representations.append(instance)
    #         # representations.append(instance)

    # return {
    #     "is_trained": index.is_trained,
    #     "ntotal": index.ntotal,
    #     "data": representations,
    #     "index": index,
    # }


def save_faiss_index(index, index_filename):
    faiss.write_index(index, index_filename)


def load_faiss_index(index_filename):
    print("Loading index", index_filename)
    return faiss.read_index(index_filename)


async def build_faiss_index(embeddings, index_filename="faces.index"):
    print("BUILDING >>>>>>>>>>>>")
    embeddings_np = np.array(embeddings, dtype=np.float32)
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)

    # nlist = 25
    # quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # index.nprobe = 10

    # assert not index.is_trained
    # index.train(embeddings_np)
    # assert index.is_trained

    index.add(embeddings_np)

    save_faiss_index(index, index_filename)

    return index


async def search_faces_by_image(img, index_filename=None):
    db = Prisma()
    await db.connect()

    start_time = time.time()

    temp_embeds = await db.face.find_many()

    # if index_filename:
    #     # Try to load the index from disk
    #     try:
    #         index = load_faiss_index(index_filename)
    #     except Exception as e:
    #         print("Failed to load index from disk:", e)
    #         index_filename = None
    # else:
    #     embeddings = [face.embedding for face in temp_embeds]
    #     index = await build_faiss_index(embeddings, index_filename)

    # temp_embeds = await db.face.find_many()

    embeddings = [face.embedding for face in temp_embeds]

    # print(type(temp_embeds[0]))

    # np.random.seed(1234)
    # embeddings_np = np.array(embeddings, dtype=np.float32)
    # d = embeddings_np.shape[1]

    # index = faiss.IndexFlatL2(d)

    # nlist = 25
    # quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # assert not index.is_trained
    # index.train(embeddings_np)
    # assert index.is_trained

    # index.add(embeddings_np)
    # print(index)

    embeddings_np = np.array(embeddings, dtype=np.float32)
    d = embeddings_np.shape[1]
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)

    # start_time = time.time()

    img_obj = DeepFace.extract_faces(img, detector_backend="retinaface")

    query_embed = DeepFace.represent(
        img_obj[0]["face"],
        detector_backend="skip",
        # detector_backend="retinaface",
        # model_name="Facenet",
    )
    query_embedding = query_embed[0]["embedding"]

    query_embedding_np = np.array(query_embedding, dtype=np.float32)
    # print(type(query_embedding_np))
    distances, indices = index.search(query_embedding_np.reshape(1, -1), k=index.ntotal)

    neighbors_with_distances = list(zip(distances[0], indices[0]))

    sorted_neighbors = sorted(neighbors_with_distances, key=lambda x: x[0])

    sorted_distances = [distance for distance, _ in sorted_neighbors]

    sorted_indices = [index for _, index in sorted_neighbors]

    images = []

    # print(sorted_indices)

    # for index in sorted_indices:
    #     image = temp_embeds[index].image_url
    #     if image not in images:
    #         images.append(image)

    for distance, index in sorted_neighbors:
        # if distance < 0.18:
        image = temp_embeds[index].image_url, 25 * (4 - distance)
        if image not in images:
            images.append(image)

    end_time = time.time()

    return images, end_time - start_time, sorted_distances
