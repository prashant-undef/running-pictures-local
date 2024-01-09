import shutil
import os
from deepface import DeepFace

# from natsort import natsorted
# import matplotlib.pyplot as plt
# import pandas as pd
# from deepface.commons import distance as dst

# import os

# import cv2
# from index_faces import index_faces, search_faces_by_image, find
from index_faces import find

# import faiss
# import numpy as np

metrics = ["cosine", "euclidean", "euclidean_l2"]

# img1_path = "dataset/couple.jpg"
# img2_path = "dataset2/Runathon-of-Hope-2131.jpg"

# img1 = DeepFace.extract_faces(img1_path)
# img2 = DeepFace.detectFace(img2_path)

import asyncio

# from prisma import Prisma


async def main() -> None:
    # result = await index_faces(
    #     db_path="dataset2/",
    #     # "dataset/",
    #     detector_backend="retinaface",
    #     # model_name="Facenet",
    #     distance_metric="euclidean_l2",
    #     enforce_detection=False,
    # )

    # print(f"{len(result)} faces found")

    # result = await search_faces_by_image(
    #     img="dataset2/img.jpg",
    #     index_filename="faces.index",
    # )

    df = find(
        "dataset2/img.jpg",
        db_path="dataset2/",
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False,
        distance_metric=metrics[0],
        silent=True
        # enforce_detection=False,
    )

    # df["formatted_distance"] = df[]["ArcFace_cosine"].apply(lambda x: f"{x:.6f}")

    # Create a list of tuples
    # result_tuples = list(
    #     df[["identity", "formatted_distance"]].itertuples(index=False, name=None)
    # )
    print(df[0])
    file_paths = df[0]["identity"].values
    results_folder = "results"

    try:
        shutil.rmtree(results_folder)
        print(f"Deleted existing {results_folder} folder.")
    except FileNotFoundError:
        print(f"{results_folder} folder not found.")

    os.makedirs(results_folder, exist_ok=True)

    for index, file_path in enumerate(file_paths):
        new_filename = f"{index + 1}_{os.path.basename(file_path)}"
        destination_path = os.path.join(results_folder, new_filename)

        shutil.copy(file_path, destination_path)

    print(f"Images copied to {results_folder} folder.")


# print(df[0]["ArcFace_cosine"].apply(lambda x: f"{x:.100f}").tolist())
# result = [(res[0]["identity"], res[0]["ArcFace_cosine"]) for res in results]
# print(result)
# print(result[0:2])
# print("distance: ", result[-1])


if __name__ == "__main__":
    asyncio.run(main())

# for img_index in sorted_indices:
#     print(result["data"][img_index]["image"])

# print(embeddings)
# np.random.seed(1234)
# index = faiss.IndexFlatL2(len(embeddings[0]))
# database_embeddings_np = np.array(embeddings, dtype=np.float32)
# index.add(database_embeddings_np)
# print(index.ntotal)

# query_embed = DeepFace.represent(
#     img1_path,
# )
# query_embedding = query_embed[0]["embedding"]

# query_embedding_np = np.array(query_embedding, dtype=np.float32)
# k = 10
# # print(type(query_embedding_np))
# distances, indices = index.search(query_embedding_np.reshape(1, -1), k)

# neighbors_with_distances = list(zip(distances[0], indices[0]))

# sorted_neighbors = sorted(neighbors_with_distances, key=lambda x: x[0])

# sorted_indices = [index for _, index in sorted_neighbors]

# print(sorted_indices)

# plt.imshow(img1)
# img = DeepFace.represent("dataset2/img.jpg", detector_backend="retinaface")
# # model_name = "Facenet"
# img2 = DeepFace.represent(
#     "dataset2/Runathon-of-Hope-2131.jpg", detector_backend="retinaface"
# )

# detected_faces = DeepFace.extract_faces(img2_path, detector_backend="mtcnn")
# # print("img_face", type(detected_faces[0]["face"]))

# detected_faces_folder_path = "detected/"
# i = 1

# # for detected_face in detected_faces:
# #     org_image_file = f"img{i}.jpg"
# #     new_image_path = os.path.join(detected_faces_folder_path, org_image_file)
# #     face_temp = detected_face["face"][:, :, ::-1]
# #     face_temp = face_temp * 255
# #     cv2.imwrite(new_image_path, face_temp)
# #     print("Detected face saved in:", new_image_path)
# #     i += 1
# # org_images.append(org_image)

# target_objs = DeepFace.extract_faces(
#     img2_path,
#     # target_size=target_size,
#     # target_size=(224, 224),
#     detector_backend="retinaface",
#     # grayscale=False,
#     # enforce_detection=enforce_detection,
#     # align=align,
# )


# # print(target_objs)

# resp_obj = []

# for target_img, target_region, _ in target_objs:
#     print(target_img)
#     print(target_region)
#     target_embedding_obj = DeepFace.represent(
#         img_path=target_img,
#         # detector_backend="ssd",
#         enforce_detection=False,
#         # model_name=model_name,
#         # enforce_detection=enforce_detection,
#         detector_backend="skip",
#         # align=align,
#         # normalization=normalization,
#     )

#     target_representation = target_embedding_obj[0]["embedding"]
#     print(target_embedding_obj)
#     # result_df = df.copy()  # df will be filtered in each img
#     # result_df["source_x"] = target_region["x"]
#     # result_df["source_y"] = target_region["y"]
#     # result_df["source_w"] = target_region["w"]
#     # result_df["source_h"] = target_region["h"]

#     # distances = []
#     # for index, instance in df.iterrows():
#     #     source_representation = instance[f"{model_name}_representation"]

#     #     if distance_metric == "cosine":
#     #         distance = dst.findCosineDistance(
#     #             source_representation, target_representation
#     #         )
#     #     elif distance_metric == "euclidean":
#     #         distance = dst.findEuclideanDistance(
#     #             source_representation, target_representation
#     #         )
#     #     elif distance_metric == "euclidean_l2":
#     #         distance = dst.findEuclideanDistance(
#     #             dst.l2_normalize(source_representation),
#     #             dst.l2_normalize(target_representation),
#     #         )
#     #     else:
#     #         raise ValueError(f"invalid distance metric passes - {distance_metric}")

#     #     distances.append(distance)

#     #     # ---------------------------

#     # result_df[f"{model_name}_{distance_metric}"] = distances

#     # threshold = dst.findThreshold(model_name, distance_metric)
#     # result_df = result_df.drop(columns=[f"{model_name}_representation"])
#     # result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
#     # result_df = result_df.sort_values(
#     #     by=[f"{model_name}_{distance_metric}"], ascending=True
#     # ).reset_index(drop=True)

#     # resp_obj.append(result_df)

# print(resp_obj)


# # # distance = dst.findEuclideanDistance(
# # #     dst.l2_normalize(img[0]["embedding"]), dst.l2_normalize(img2[0]["embedding"])
# # # )
# # # threshold = dst.findThreshold("ArcFace", "euclidean_l2")

# # # # identinfied = True
# # # print(distance)
# # # print(threshold)
# # # if distance <= threshold:
# # #     print(True)
# # # else:
# # #     print(False)

# # # print(identified)

# # # df = DeepFace.verify(
# # #     img,
# # #     img2
# # #     # db_path="dataset2/",
# # #     # model_name="ArcFace",
# # #     # distance_metric=metrics[2],
# # # )

# # # print(resp)
# # # print(img1)


# # # print(f"Similarity Score: {result['similarity']}")
# # # print(f"Image Path: {result['img_path']}")
# # # df[0]
# # # img_paths = map(df[0]["identity"])

# # # df[0].style.hide(names=True)
# # # image_paths = natsorted(df[0]["identity"].tolist())

# # # print(image_paths)
