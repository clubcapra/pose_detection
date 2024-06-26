VECTOR_COUPLES = [[0, 1],
                [1, 2],
                [3, 4],
                [4, 5]]

KEYPOINT_COUPLES = [[1, 2],
                    [2, 3],
                    [3, 4],
                    [1, 5],
                    [5, 6],
                    [6, 7]]

KEYPOINT_OF_INTEREST = [1, 2, 3, 4, 5, 6, 7]

FACE_KEYPOINTS = [0, 14, 15]

ERROR_MARGIN = 30

TPOSE_ANGLES = [0, 10, 0, 10]

SKYWARD_ANGLES = [90, 0, 90, 0]

BUCKET_ANGLES = [0, 105, 0, 105]

DATASET_PATH = "./dataset"

PREDICTION_OUTPUT_DICT = {
        0: "NO POSE",
        1: "T-POSE",
        2: "BUCKET",
        3: "SKYWARD" 
}

POSE_DICT = {
        "NO POSE": 0,
        "T-POSE": 1,
        "BUCKET": 2,
        "SKYWARD": 3
}

STATE_DICT = {
    POSE_DICT["T-POSE"]: 'IDLE',
    POSE_DICT["BUCKET"]: 'FOLLOW',
    POSE_DICT["SKYWARD"]: 'RETRACE'
}
