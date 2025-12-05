#!/usr/bin/env python3

import cv2
import numpy as np
import math
import time

def skin_mask_ycrcb(frame):
    # convert to YCrCb and threshold for skin
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    # refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    return mask

def find_hand_contours(mask, min_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            hands.append((area, c))
    # sort decresing area (largest first)
    hands.sort(key=lambda x: x[0], reverse=True)
    return [c for _,c in hands]

def count_fingers_from_contour(contour, drawing):
    # bounding rect
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    # find bounding box & center (approx wrist)
    x,y,w,h = cv2.boundingRect(contour)
    wrist_y = y + h  # bottom as wrist approximated
    finger_points = []
    valid_defects = 0

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end   = tuple(contour[e][0])
        far   = tuple(contour[f][0])
        # geometry
        a = np.linalg.norm(np.array(start)-np.array(far))
        b = np.linalg.norm(np.array(end)-np.array(far))
        c = np.linalg.norm(np.array(start)-np.array(end)) + 1e-6
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (a*a + b*b - c*c) / (2*a*b+1e-6)))))
        # depth d is multiplied by something depending on image scale
        if angle < 90 and d > 10000:
            valid_defects += 1
            finger_points.append(start)
            finger_points.append(end)
            cv2.circle(drawing, far, 5, (0,255,0), -1)

    # number of fingers = defects + 1 but clamp to 5
    fingers = min(5, valid_defects + 1) if valid_defects > 0 else 0

    # try to refine: detect fingertip by scanning contour points above wrist line
    # This helps in some orientations
    contour_pts = contour.reshape(-1,2)
    top_pts = [tuple(p) for p in contour_pts if p[1] < wrist_y - h*0.15]
    # cluster top points to unique finger tips using distance threshold
    tips = []
    for p in top_pts:
        if not tips:
            tips.append(p)
        else:
            dmin = min([math.hypot(p[0]-q[0], p[1]-q[1]) for q in tips])
            if dmin > 30:
                tips.append(p)
    # choose up to 5
    tips = sorted(tips, key=lambda t: t[0])[:5]
    # if tips found, prefer their count when different and sensible
    if 0 < len(tips) <=5:
        fingers = len(tips)

    return fingers

def annotate_hand_info(img, cnts):
    vis = img.copy()
    info = []
    for idx, cnt in enumerate(cnts[:2]):  # at most 2 hands
        x,y,w,h = cv2.boundingRect(cnt)
        roi = vis[y:y+h, x:x+w]
        # draw bounding box
        cv2.rectangle(vis, (x,y), (x+w, y+h), (200,100,0), 2)
        # count fingers
        fingers = count_fingers_from_contour(cnt, vis)
        cv2.putText(vis, f"H{idx} Fingers:{fingers}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        info.append((idx, fingers))
    return vis, info

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera")
        return
    prev = None
    last_ts = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h,w = frame.shape[:2]

        mask = skin_mask_ycrcb(frame)
        contours = find_hand_contours(mask, min_area=2500)
        vis, info = annotate_hand_info(frame, contours)

        # overlay total
        total = sum([f for (_,f) in info]) if info else 0
        cv2.putText(vis, f"Total fingers: {total}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # show mask small
        mask_small = cv2.resize(mask, (200,150))
        vis[0:150, w-200:w] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Hand Counter - OpenCV", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        # tuning keys
        if key == ord('s'):
            cv2.imwrite("snapshot.jpg", frame)
            print("Snapshot saved.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
