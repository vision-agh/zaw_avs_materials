from os.path import join
import cv2
import numpy as np

############################### Funkcje #################################

def detect(net, img):
    size = img.shape
    height = size[0]
    width = size[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes

def filter_boxes(boxes):
    all_paired_boxes = list()
    for ii, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        center_x1 = x1 + int(w1 / 2)
        center_y1 = y1 + int(h1 / 2)
        to_connect = [ii]
        for jj, box2 in enumerate(boxes):
            if jj != ii:
                x2, y2, w2, h2 = box2
                center_x2 = x2 + int(w2 / 2)
                center_y2 = y2 + int(h2 / 2)
                if abs(center_x2 - center_x1) < 10 and abs(center_y2 - center_y1) < 10:
                    to_connect.append(jj)
        all_paired_boxes.append(to_connect)
    all_paired_boxes = sorted(all_paired_boxes, key=lambda x: len(x), reverse=True)
    all_paired = list()
    final_boxes = list()
    for conn in all_paired_boxes:
        if all([a not in all_paired for a in conn]):
            for a in conn:
                all_paired.append(a)
            final_boxes.append(conn)
    out_boxes = [[int(sum([boxes[i][a] for i in elem]) / len(elem)) for a in range(4)] for elem in final_boxes]
    return out_boxes

def IoU(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    left = max([x1, x2])
    right = min([x1+w1, x2+w2])
    top = max([y1, y2])
    bottom = min([y1+h1, y2+h2])
    area1 = max([(right - left), 0]) * max([(bottom - top), 0])
    area2 = (w1 * h1) + (w2 * h2) - area1
    IoU = area1/area2
    return IoU

############# METHOD ###############
# Wybierz metodę fuzji
#FUSION = "LATE"
FUSION = "EARLY"

############# TODO0 ###############
# Ustaw ścieżki
test_rgb = "test_rgb"  # Ścieżka do foldeu test_rgb
test_thermal = "test_thermal"  # Ścieżka do foldeu test_thermal
###################################

net_fus = None
net_therm = None
net_rgb = None
if FUSION == "EARLY":
    net_fus = cv2.dnn.readNet('yolov3_training_last_f.weights', 'yolov3_testing_f.cfg')
if FUSION == "LATE":
    net_therm = cv2.dnn.readNet('yolov3_training_last_t.weights', 'yolov3_testing_t.cfg')
    net_rgb = cv2.dnn.readNet('yolov3_training_last_c.weights', 'yolov3_testing_c.cfg')

for i in range(200, 300): # można zmieniać range do 518
    path_rgb = join(test_rgb, f"img{i}.png")
    path_thermal = join(test_thermal, f"img{i}.png")
    img_rgb = cv2.imread(path_rgb)
    img_thermal = cv2.imread(path_thermal)
    img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
    out_img = None
    boxes = None
    if FUSION == "EARLY":
        ############ TODO1 ##################
        # Połącz RGB z Thermal podążając za instrukcją
        # 1. Utwórz nową ramkę (numpy array) o wymiarach zdjęcia RGB i nazwij ją "new_fus"
        # 2. Skopiuj dwa pierwsze kanały z RGB (img_rgb[:, :, :2]) do dwóch pierwszych kanałów nowej ramki (new_fus[:, :, :2])
        # 3. Wartość trzeciego kanału nowej ramki to maksimum z wartości trzeciego kanału RGB i obrazu z termowizji (jednokanałowego)
        #    Użyj w tym celu np.maximum(a, b). Gdzie a i b to 3 kanał RGB oraz termowizja
        # 4. Zrzutuj "new_fus" na "uint8" (new_fus.astype("uint8"))
        new_fus = None

        ####################################
        out_img = new_fus
        boxes = detect(net_fus, new_fus)
    if FUSION == "LATE":
        out_img = img_rgb
        Rect1 = detect(net_therm, img_thermal)
        Rect2 = detect(net_rgb, img_rgb)
        ############ TODO2 ##################
        # "Rect1" i "Rect2" mają format [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
        # 1. Utwórz listę "boxes_iou". Iterując w podwójnej pętli po "Rect1" i "Rect2" sprawdzaj wartość IoU
        # poszczególnych prostokątów zapisanych w tych listach (użyj funkcji IoU(), która jest zdefiniowana powyżej
        # i jako argumenty przyjmuje dwa prostokąty otaczające). Jeśli wartoś IoU dla danej pary jest większa od 0,
        # to do "boxes_iou" appenduj listę składającą się z krotki (zawierającej indeksy aktualnie przetwarzanych
        # prostokątów otaczających) oraz z wyliczonej dla nich wartości IoU. Przykład: W danej iteracji podwójnej
        # pętli dotarliśmy do 3 prostokąta z "Rect1" i 4 prostokąta z "Rect2". Ich wspólna wartość IoU to 0.55.
        # Do tablicy "boxes_iou" dodajemy więc listę [(3, 4), 0.55].
        # 2. Następnie posortuj "boxes_iou" malejąco po wartości IoU. Użyj do tego funkcji sorted() z parametrami
        # key=lambda a: a[1] oraz reverse=True.
        # 3. Utwórz puste listy "Rect1_paired", "Rect2_paired" i "paired_boxes".
        # 4. Utwórz pętlę po elementach "boxes_iou". W każdej iteracji z aktualnie przetwarzanego elementu wyciągnij
        # krotkę z indeksami(elem[0]) oraz wartość IoU(elem[1]). Jeśli pierwszy indeks z krotki nie występuje
        # w liście "Rect1_paired" oraz drugi element z krotki nie występuje w liście "Rect2_paired", to do
        # "paired_boxes" appendujemy krotkę z indeksami(elem[0]), a do list "Rect1_paired" i "Rect2_paired" appendujemy
        # odpowiednie indeksy z krotki (pierwszy do pierwszej z list i drugi do drugiej).
        # W taki sposób otrzymaliśmy listę "paired_boxes", która zawiera pary indeksów prostokątów z list "Rect1"
        # i "Rect2", które należy połączyć (uśrednić ich elementy), co zostanie opisane w punkcie 5.
        # 5. Na koniec tworzymy pustą listę "boxes". Iterujemy po krotkach w "paired_boxes", wyciągamy z "Rect1"
        # prostokąt o indeksie zapisanym jako pierwszy element krotki, a z "Rect2" wyciągamy prostokąt o indeksie
        # zapisanym jako drugi element krotki. Prostokąty mają postać listy 4 elementowej ([x1, y1, w1, h1]).
        # Mając 2 prostokąty, czyli 2 listy 4 elementowe (nazwijmy je "r1" i "r2") tworzymy jedną nową listę
        # 4 elementową (nazwijmy ją "avg_r"), której elemnty są uśrednieniem elementów z obu list z prostokątami
        # (pamiętamy, żeby po wyliczeniu średniej, wynik zrzutować na int, avg_r[0] = int((r1[0]/r2[0])/2) i tak dla
        # wszystkich 4 elementów. Na koniec appendujemy "avg_r" do listy "boxes". W taki sposób format listy "boxes"
        # będzie taki sam jak format list "Rect1" i "Rect2".
        boxes = None

        ######################################
    out_boxes = filter_boxes(boxes)
    for box in out_boxes:
        x, y, w, h = box
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.imshow('Image', out_img)
    cv2.waitKey(10)
