import cv2
import mediapipe as mp

def mark_grip_points(image_path, output_path):
    # 使用 MediaPipe 手部模块
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
        return
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        print("Hand landmarks detected.")
        for hand_landmarks in results.multi_hand_landmarks:
            # 遍历手部的关键点并进行可视化
            for landmark in hand_landmarks.landmark:
                # 关键点坐标
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 绘制每个关键点
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # 用绿色标记
            
            # 绘制手部骨架
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 识别并标注握住杯柄的点
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # 获取拇指和食指的坐标
        h, w, _ = image.shape
        thumb_tip_coord = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_tip_coord = (int(index_tip.x * w), int(index_tip.y * h))

        # 标注握住杯柄的点
        cv2.circle(image, thumb_tip_coord, 10, (255, 0, 0), -1)  # 用红色标记拇指
        cv2.circle(image, index_tip_coord, 10, (255, 0, 0), -1)  # 用红色标记食指

        # 输出坐标
        print(f"Thumb tip coordinates: {thumb_tip_coord}")
        print(f"Index tip coordinates: {index_tip_coord}")
        
    else:
        print("No hands detected.")
    
    # 显示图片
    cv2.imshow("Hand Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存标注好的图片
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")

image_path = "/home/wts/code/hand_object_detector/images/000030_color.png"  # 输入的图片路径
output_path = "/home/wts/code/hand_object_detector/images/output_image.jpg"  # 输出的图片路径

mark_grip_points(image_path, output_path)