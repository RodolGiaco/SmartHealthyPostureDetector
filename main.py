import cv2 as cv
import numpy as np
import json
from mp_utils import pose_hands
from neural_network import gesture_recognition
from instructions import gesture_instructions  
from instructions import gesture_buffer
from gui import ThirdPersonGUI  # Importa la clase ThirdPersonGUI desde gui

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Configuración de valores predeterminados si faltan
    hand_tracking_confidence = config['constants']['hands'].get('min_hand_tracking_confidence', 0.5)
    hand_presence_confidence = config['constants']['hands'].get('min_hand_presence_confidence', 0.5)

    # Inicializa ThirdPersonGUI desde gui.py
    tp_gui = ThirdPersonGUI(config['constants']['gui']['hand_window_height'], config['constants']['gui']['hand_window_width'])

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    # Inicialización del detector de manos
    hand_pose_detection = pose_hands.HandPoseDetection(
        hand_region_window_width=config['constants']['gui']['hand_window_width'],
        hand_region_window_height=config['constants']['gui']['hand_window_height'],
        pose_model_asset_path=config['model_paths']['pose_landmarker'],
        min_pose_detection_confidence=config['constants']['pose']['min_pose_detection_confidence'],
        min_pose_presence_confidence=config['constants']['pose']['min_pose_presence_confidence'],
        min_pose_tracking_confidence=config['constants']['pose']['min_tracking_confidence'],
        safe_zone=config['constants']['pose']['safe_zone'],
        hand_model_asset_path=config['model_paths']['hand_landmarker'],
        min_hand_presence_confidence=hand_presence_confidence,
        min_hand_tracking_confidence=hand_tracking_confidence,
    )

    # Inicialización del reconocedor de gestos
    gesture_recognizer = gesture_recognition.GestureRecognizer(
        model_path=config['model_paths']['gesture_recogniser'],
        label_path=config['model_paths']['keypoint_classifier_labels']
    )

    # Inicialización del buffer de gestos
    buffer = gesture_buffer.GestureBuffer(buffer_len=config['constants']['buffer_length'])

    # Inicialización de las instrucciones
    instructions = gesture_instructions.Instructions(
        following=config['initial_options']['following'],
        speed=config['constants']['speed']
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar la imagen")
            break

        # Proceso de detección de manos
        pose_result = hand_pose_detection.extract_pose(frame)
        right_hand_roi = hand_pose_detection.extract_right_hand_roi(frame)
        main_window_image = hand_pose_detection.draw_pose(frame)
        hand_result = hand_pose_detection.extract_hands(right_hand_roi)
        hand_window_image = hand_pose_detection.draw_hands(right_hand_roi)

        # Reconocimiento de gestos
        gesture_id, _ = gesture_recognizer.recognize_gesture(hand_result, right_hand_roi)
        gesture_name = gesture_recognizer.translate_gesture_id_to_name(gesture_id)
        buffer.add_gesture(gesture_id)
        gesture = buffer.get_gesture()

        type_move, move = instructions.calculate_move(gesture, pose_result, frame)

        tp_gui.update_camera_window(main_window_image)
        tp_gui.update_hand_window(hand_window_image)
        tp_gui.update_info_window(instructions.get_follow_state(), move, 100, gesture_name)
        tp_gui.show_window()

        # Mover ventanas para que no se encimen
        cv.moveWindow("ThirdPerson", 0, 0)  # Ventana principal en la esquina superior izquierda
        cv.moveWindow("Info", 640, 0)       # Ventana de información a la derecha de la principal
        cv.moveWindow("hand", 640, 360)     # Ventana de mano debajo de la ventana de información

        key = tp_gui.getKey()
        if key == ord('q'):
            break

    cap.release()
    hand_pose_detection.close()
    tp_gui.close()

if __name__ == "__main__":
    main()
