import streamlit as st
from imutils import paths
import face_recognition
import glob
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from imutils import build_montages

# UPLOAD the images here to test
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

if uploaded_files:
    data = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": uploaded_file.name, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    np_data = np.array(data)
    np_encodings = [item["encoding"] for item in np_data]

    st.write("[INFO] Clustering...")
    cluster = DBSCAN(metric="euclidean", n_jobs=-1)
    cluster.fit(np_encodings)

    labelIDs = np.unique(cluster.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    st.write("[INFO] Number of unique faces: {}".format(numUniqueFaces))
    st.write("[INFO] Label IDs: {}, where [-1] refers to unidentified faces or outliers".format(labelIDs))

    for labelID in labelIDs:
        st.write("[INFO] Faces for face ID: {}".format(labelID))
        idxs = np.where(cluster.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(15, len(idxs)), replace=False)
        faces = []

        for i in idxs:
            current_image = cv2.imread(np_data[i]["imagePath"])
            (top, right, bottom, left) = np_data[i]["loc"]
            current_face = current_image[top:bottom, left:right]
            current_face = cv2.resize(current_face, (96, 96))
            faces.append(current_face)

        montage = build_montages(faces, (96, 96), (3, 3))[0]
        montage_rgb = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)

        st.image(montage_rgb, channels="RGB")
