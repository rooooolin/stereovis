import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)



def imshow(frame):
    cv2.imshow("cap", frame)
    return cv2.waitKey(100)

ret, frame = cap.read()
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    ret1 = imshow(frame)
    print("frame_id:", count)
    cv2.imwrite("./data/captest/" + str(count) + ".jpg",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
