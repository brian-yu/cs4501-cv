import cv2
import numpy as np
import sys
import pickle
import random

'''
WIDTH: 79.5 cm
HEIGHT: 52.5cm

aspect ratio: 79.5/52.5
'''

RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

# filename = 'IMG_4087.jpg'
# filename = 'IMG_4048_blank_deskewed.jpg'
filename = sys.argv[1]
image = cv2.imread(filename)

if 'deskewed' not in filename:
    scale_percent = 25 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(dim)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

mouse_coords = []

circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
color_idx = 0

tracks = pickle.load( open( "tracks.p", "rb" ) )

def create_point(x, y):
    global mouse_coords, image, circle_colors, color_idx, tracks
    color = circle_colors[color_idx]
    color_idx = (color_idx + 1) % len(circle_colors)

    cv2.circle(image,(x,y),2,color,-1)
    mouse_coords.append([x, y])
    cv2.imshow('image',image)

def delete_point():
    global mouse_coords, image, circle_colors, color_idx, tracks
    last_x, last_y = mouse_coords.pop()

    color_idx = (color_idx - 1) % len(circle_colors)

    cv2.circle(image,(last_x, last_y), 2, (0,0,0), -1)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       # cv2.circle(image, (x,y),100,(255,0,0),-1)
       print('x = %d, y = %d'%(x, y))
       create_point(x, y)


def deskew(image):
    global mouse_coords, filename
    print(mouse_coords[-4:])

    pts_src = np.array(mouse_coords[-4:])

    pts_dst = np.array([[0, 0], [WIDTH-1, 0], [WIDTH-1, HEIGHT-1], [0, HEIGHT-1]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (WIDTH, HEIGHT))

    prefix = filename.split(".")[0]
    cv2.imwrite(f"{prefix}_deskewed.jpg", im_out)

    return im_out

def draw_tracks(image, tracks):
    for track in tracks:

        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))

        for box in track:
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1,1,2))

            # deskew(image, box, TRAIN_WIDTH, TRAIN_HEIGHT)

            cv2.polylines(image,[pts],True,color, 3)

cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)

while True:
    cv2.imshow('image',image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    elif k == ord('c'):
        print("CROP")
        image = deskew(image)
        draw_tracks(image, tracks)
    elif k == ord('d'):
        # print(mouseX,mouseY)
        print("DELETE")
        delete_point()

    elif k == ord('s'):
        # print(mouseX,mouseY)
        print("saving track")
        save_track()

        

cv2.destroyAllWindows()



"""
tracks = [[[[180, 150], [226, 147], [224, 121], [178, 125]], [[230, 145], [279, 140], [275, 116], [226, 121]], [[282, 140], [330, 131], [326, 109], [278, 113]]], [[[368, 124], [412, 106], [404, 82], [357, 102]], [[414, 105], [463, 94], [456, 71], [411, 79]], [[466, 94], [514, 93], [513, 68], [463, 68]], [[519, 91], [563, 97], [570, 73], [518, 66]], [[572, 98], [617, 112], [625, 86], [574, 73]], [[621, 112], [663, 132], [677, 109], [629, 88]]], [[[709, 146], [756, 158], [762, 134], [713, 123]], [[762, 159], [807, 171], [812, 145], [766, 132]], [[813, 173], [859, 183], [864, 156], [815, 144]], [[862, 183], [906, 194], [912, 167], [866, 156]], [[912, 194], [955, 205], [963, 175], [918, 166]], [[963, 205], [1005, 212], [1013, 187], [968, 175]]], [[[1047, 209], [1088, 179], [1074, 153], [1032, 181]], [[1090, 179], [1134, 156], [1122, 129], [1076, 146]], [[1137, 155], [1183, 138], [1175, 111], [1125, 124]], [[1184, 137], [1234, 126], [1227, 98], [1178, 106]], [[1235, 127], [1285, 123], [1285, 97], [1233, 96]]], [[[1325, 124], [1368, 157], [1384, 137], [1342, 104]], [[1369, 157], [1410, 188], [1424, 169], [1384, 136]]], [[[1313, 141], [1352, 173], [1366, 155], [1326, 125]], [[1356, 175], [1393, 203], [1408, 188], [1367, 157]]], [[[1256, 155], [1303, 137], [1294, 118], [1246, 134]], [[1216, 188], [1254, 158], [1239, 138], [1198, 168]], [[1191, 235], [1216, 193], [1192, 176], [1168, 219]]], [[[1290, 147], [1314, 144], [1324, 189], [1296, 192]], [[1298, 199], [1324, 196], [1334, 238], [1302, 242]], [[1304, 248], [1337, 245], [1340, 285], [1311, 289]]], [[[1393, 206], [1416, 219], [1389, 259], [1368, 245]], [[1365, 250], [1387, 262], [1360, 305], [1340, 292]]], [[[1436, 231], [1415, 217], [1388, 261], [1411, 270]], [[1407, 276], [1386, 261], [1361, 302], [1380, 317]]], [[[1050, 227], [1102, 236], [1104, 211], [1059, 203]], [[1109, 212], [1106, 235], [1152, 244], [1156, 219]]], [[[135, 162], [159, 162], [156, 215], [136, 215]]], [[[157, 163], [182, 163], [179, 212], [156, 213]]], [[[171, 213], [218, 213], [222, 236], [173, 236]], [[224, 210], [227, 233], [279, 228], [273, 206]], [[286, 226], [330, 196], [309, 177], [273, 202]], [[335, 189], [356, 141], [332, 132], [309, 171]]], [[[354, 141], [375, 124], [403, 161], [381, 181]], [[383, 186], [409, 164], [438, 200], [413, 225]], [[414, 229], [443, 207], [470, 241], [443, 268]], [[446, 270], [473, 248], [503, 281], [480, 303]]], [[[503, 282], [520, 300], [558, 270], [539, 249]], [[541, 244], [562, 267], [596, 237], [575, 213]], [[579, 208], [613, 173], [634, 195], [597, 233]], [[638, 196], [674, 161], [652, 137], [614, 171]]], [[[682, 160], [701, 140], [735, 171], [715, 196]], [[716, 198], [739, 174], [775, 209], [751, 233]], [[755, 236], [776, 214], [812, 247], [790, 269]], [[791, 272], [815, 248], [849, 280], [825, 302]]], [[[864, 271], [875, 292], [919, 277], [911, 254]], [[923, 275], [970, 257], [958, 233], [914, 247]], [[974, 255], [1018, 237], [1005, 214], [962, 228]]], [[[129, 230], [148, 240], [130, 290], [108, 278]]], [[[128, 291], [147, 243], [166, 253], [145, 297]]], [[[169, 259], [215, 269], [222, 248], [173, 235]], [[220, 272], [268, 282], [272, 254], [227, 246]], [[272, 283], [319, 294], [323, 267], [276, 254]], [[322, 292], [365, 304], [374, 275], [325, 266]], [[369, 306], [419, 317], [424, 287], [374, 277]], [[425, 315], [467, 327], [474, 300], [427, 288]]], [[[517, 324], [568, 326], [566, 299], [515, 299]], [[571, 325], [621, 327], [621, 301], [571, 297]], [[624, 325], [673, 327], [673, 299], [623, 297]], [[676, 326], [722, 326], [722, 298], [673, 297]], [[727, 325], [777, 328], [776, 296], [728, 297]], [[781, 326], [825, 325], [826, 299], [777, 298]]], [[[865, 317], [917, 310], [911, 288], [862, 291]], [[917, 314], [965, 303], [963, 275], [913, 283]], [[968, 303], [1017, 294], [1009, 263], [965, 270]], [[1020, 294], [1067, 285], [1063, 254], [1015, 260]], [[1071, 284], [1119, 273], [1113, 250], [1065, 253]], [[1122, 273], [1169, 261], [1162, 245], [1119, 246]]], [[[447, 368], [470, 381], [494, 338], [470, 324]], [[467, 385], [444, 369], [420, 409], [444, 425]], [[417, 412], [446, 430], [419, 470], [392, 456]]], [[[493, 335], [518, 325], [540, 369], [505, 379]], [[510, 386], [540, 373], [558, 417], [527, 429]], [[530, 434], [560, 424], [579, 465], [543, 477]], [[551, 483], [582, 470], [595, 513], [568, 525]]], [[[138, 321], [183, 330], [189, 301], [142, 292]], [[186, 332], [196, 301], [243, 320], [224, 350]], [[231, 349], [245, 323], [290, 346], [270, 374]], [[277, 377], [297, 350], [332, 382], [305, 408]], [[312, 410], [336, 387], [370, 424], [342, 446]], [[348, 448], [375, 429], [398, 469], [371, 486]]], [[[525, 350], [534, 321], [585, 335], [571, 366]], [[576, 368], [589, 336], [631, 352], [624, 387]], [[631, 387], [636, 357], [682, 376], [669, 407]], [[674, 407], [683, 374], [730, 392], [720, 423]], [[722, 427], [733, 391], [778, 411], [766, 439]]], [[[778, 416], [802, 427], [820, 377], [795, 367]], [[820, 376], [838, 326], [815, 322], [797, 364]]], [[[802, 426], [818, 432], [841, 385], [819, 376]], [[819, 374], [842, 383], [856, 337], [838, 330]]], [[[854, 336], [897, 360], [909, 333], [863, 312]], [[899, 364], [948, 377], [954, 350], [910, 334]], [[957, 350], [950, 376], [997, 388], [1002, 359]]], [[[1001, 362], [1021, 382], [1055, 349], [1030, 330]], [[1034, 323], [1061, 348], [1098, 320], [1079, 294]], [[1090, 289], [1099, 320], [1148, 305], [1133, 276]], [[1136, 278], [1154, 301], [1191, 269], [1174, 252]]], [[[1187, 254], [1218, 254], [1220, 300], [1188, 300]], [[1190, 309], [1221, 308], [1221, 350], [1196, 352]]], [[[115, 314], [97, 358], [75, 347], [94, 302]], [[95, 366], [67, 358], [61, 406], [85, 412]], [[86, 417], [57, 422], [59, 467], [85, 467]], [[87, 476], [56, 476], [59, 522], [91, 522]], [[92, 527], [62, 534], [73, 577], [100, 573]]], [[[115, 315], [137, 325], [119, 366], [98, 359]], [[96, 367], [118, 370], [111, 416], [89, 414]], [[86, 421], [111, 421], [111, 467], [88, 466]], [[87, 475], [110, 474], [116, 519], [90, 522]], [[93, 529], [117, 522], [127, 565], [103, 572]]], [[[1306, 288], [1321, 309], [1275, 333], [1259, 312]], [[1255, 308], [1274, 332], [1229, 361], [1218, 336]]], [[[1331, 331], [1320, 311], [1274, 336], [1287, 355]], [[1274, 335], [1289, 357], [1243, 379], [1230, 361]]], [[[1199, 368], [1143, 365], [1143, 342], [1196, 344]], [[1140, 364], [1085, 372], [1080, 350], [1132, 340]], [[1084, 374], [1035, 393], [1027, 370], [1074, 352]]], [[[1193, 388], [1198, 367], [1141, 364], [1141, 384]], [[1140, 386], [1140, 365], [1085, 372], [1091, 394]], [[1086, 395], [1082, 374], [1034, 391], [1043, 411]]], [[[1004, 408], [1006, 388], [957, 380], [956, 401]], [[951, 406], [954, 380], [906, 371], [906, 397]], [[907, 398], [891, 375], [850, 398], [867, 423]], [[865, 426], [849, 403], [810, 431], [825, 451]]], [[[788, 459], [784, 434], [734, 438], [740, 466]], [[739, 467], [729, 442], [681, 455], [693, 482]], [[690, 484], [677, 459], [632, 479], [649, 505]], [[647, 509], [628, 484], [590, 511], [606, 533]]], [[[811, 510], [831, 503], [807, 457], [786, 462]]], [[[830, 505], [849, 492], [830, 447], [808, 454]]], [[[936, 486], [958, 501], [983, 459], [964, 447]], [[962, 441], [983, 456], [1009, 414], [992, 404]]], [[[958, 500], [976, 512], [1002, 468], [983, 456]], [[983, 455], [1002, 467], [1029, 427], [1008, 414]]], [[[978, 532], [1023, 508], [1012, 486], [967, 509]], [[1023, 508], [1070, 483], [1056, 461], [1011, 482]], [[1057, 457], [1072, 482], [1113, 456], [1098, 434]], [[1101, 432], [1115, 457], [1161, 431], [1147, 405]], [[1147, 404], [1162, 432], [1201, 406], [1191, 385]]], [[[1063, 556], [1086, 567], [1105, 522], [1079, 512]], [[1084, 501], [1108, 524], [1140, 486], [1119, 468]], [[1129, 458], [1141, 485], [1186, 463], [1170, 438]], [[1171, 437], [1196, 452], [1220, 411], [1198, 402]]], [[[1243, 506], [1271, 500], [1257, 454], [1228, 461]], [[1228, 458], [1258, 451], [1245, 407], [1219, 411]]], [[[1232, 401], [1244, 378], [1289, 398], [1276, 425]], [[1277, 426], [1290, 398], [1332, 418], [1326, 445]]], [[[1334, 427], [1358, 427], [1354, 377], [1330, 378]], [[1330, 378], [1355, 377], [1352, 327], [1327, 327]]], [[[1357, 427], [1380, 428], [1379, 375], [1354, 375]], [[1354, 375], [1381, 373], [1376, 325], [1353, 325]]], [[[1260, 516], [1281, 534], [1313, 495], [1296, 478]], [[1296, 475], [1314, 493], [1347, 453], [1329, 438]]], [[[1282, 533], [1298, 548], [1332, 508], [1316, 494]], [[1314, 492], [1334, 508], [1365, 468], [1348, 451]]], [[[127, 582], [119, 557], [167, 540], [175, 565]], [[175, 565], [168, 537], [215, 523], [225, 549]], [[225, 549], [216, 522], [264, 507], [275, 532]], [[275, 532], [266, 505], [313, 489], [321, 517]], [[324, 515], [315, 490], [360, 474], [371, 500]]], [[[126, 581], [134, 601], [184, 587], [175, 565]], [[175, 565], [182, 588], [231, 571], [223, 549]], [[226, 549], [234, 571], [280, 556], [271, 532]], [[273, 532], [281, 554], [329, 538], [322, 514]], [[324, 517], [332, 538], [380, 521], [372, 501]]], [[[399, 493], [399, 471], [452, 470], [453, 490]], [[458, 492], [458, 469], [512, 481], [507, 505]], [[511, 506], [522, 484], [566, 505], [560, 524]]], [[[402, 514], [402, 494], [453, 493], [453, 514]], [[453, 517], [459, 496], [506, 506], [501, 528]], [[505, 530], [514, 507], [559, 525], [550, 544]]], [[[604, 526], [603, 550], [656, 556], [654, 531]], [[656, 531], [658, 555], [709, 552], [704, 525]], [[706, 527], [713, 551], [761, 535], [749, 513]], [[749, 513], [762, 536], [811, 519], [801, 493]]], [[[602, 547], [601, 572], [651, 581], [652, 558]], [[657, 557], [659, 580], [712, 579], [707, 554]], [[713, 552], [720, 573], [769, 559], [760, 537]], [[762, 535], [773, 560], [816, 538], [810, 519]]], [[[841, 515], [892, 514], [889, 491], [841, 492]], [[894, 514], [893, 489], [940, 487], [943, 511]]], [[[843, 513], [843, 534], [892, 537], [892, 514]], [[892, 514], [894, 535], [941, 536], [942, 511]]], [[[967, 555], [973, 530], [1021, 544], [1015, 569]], [[1018, 570], [1025, 545], [1069, 562], [1064, 584]]], [[[1096, 551], [1110, 575], [1152, 545], [1138, 523]], [[1147, 517], [1155, 546], [1202, 533], [1197, 506]], [[1205, 505], [1205, 533], [1255, 536], [1258, 509]]], [[[1272, 567], [1286, 539], [1330, 565], [1315, 589]], [[1330, 567], [1349, 578], [1323, 625], [1297, 611]]], [[[1192, 631], [1235, 600], [1219, 584], [1179, 617]], [[1220, 585], [1259, 550], [1275, 567], [1235, 599]]], [[[1259, 551], [1244, 535], [1203, 563], [1221, 584]], [[1218, 586], [1201, 564], [1162, 596], [1179, 616]]], [[[1142, 618], [1157, 597], [1116, 570], [1101, 594]]], [[[1190, 627], [1190, 649], [1241, 659], [1241, 633]], [[1243, 628], [1246, 657], [1294, 653], [1289, 625]]], [[[1057, 629], [1097, 592], [1076, 573], [1039, 605]], [[1035, 605], [1053, 633], [1004, 651], [992, 625]], [[989, 627], [992, 656], [946, 657], [944, 630]]], [[[969, 534], [942, 528], [932, 575], [961, 580]], [[961, 585], [931, 577], [921, 626], [949, 632]]], [[[919, 627], [920, 653], [870, 651], [872, 626]], [[868, 624], [869, 653], [818, 653], [818, 633]]], [[[859, 539], [837, 530], [822, 578], [844, 586]], [[845, 587], [822, 579], [809, 627], [829, 634]]], [[[838, 530], [816, 523], [798, 571], [823, 578]], [[823, 579], [798, 571], [783, 619], [808, 627]]], [[[781, 618], [783, 642], [734, 644], [734, 619]], [[729, 619], [729, 643], [679, 639], [682, 615]], [[681, 613], [673, 640], [626, 625], [637, 595]], [[635, 593], [616, 618], [581, 584], [602, 566]]], [[[592, 559], [565, 559], [556, 605], [589, 608]], [[589, 610], [557, 610], [555, 657], [587, 660]]], [[[568, 564], [560, 537], [510, 551], [522, 579]], [[519, 583], [502, 555], [458, 582], [477, 608]], [[477, 611], [453, 587], [423, 623], [444, 644]], [[442, 649], [416, 631], [393, 675], [419, 691]], [[417, 694], [390, 684], [380, 730], [407, 737]]], [[[751, 667], [748, 642], [698, 645], [703, 672]], [[698, 674], [695, 646], [647, 651], [651, 680]], [[646, 682], [643, 652], [593, 654], [596, 686]]], [[[561, 687], [549, 663], [502, 682], [517, 708]], [[513, 712], [503, 684], [455, 706], [467, 733]], [[466, 734], [453, 706], [406, 730], [418, 752]]], [[[406, 516], [379, 510], [369, 559], [400, 565]], [[396, 575], [370, 562], [351, 606], [377, 618]], [[371, 631], [350, 607], [310, 638], [330, 663]]], [[[297, 671], [292, 644], [241, 653], [250, 680]], [[250, 681], [226, 667], [206, 710], [226, 724]]], [[[110, 602], [132, 592], [151, 637], [127, 645]], [[132, 655], [152, 639], [180, 679], [159, 695]], [[163, 700], [180, 680], [215, 717], [198, 733]]], [[[110, 602], [87, 610], [105, 655], [129, 648]], [[133, 654], [112, 666], [136, 707], [158, 694]], [[163, 700], [144, 717], [180, 751], [197, 735]]], [[[557, 686], [584, 688], [583, 736], [552, 735]], [[552, 739], [582, 743], [578, 789], [549, 789]]], [[[576, 786], [583, 810], [631, 794], [622, 770]], [[623, 767], [637, 792], [680, 770], [668, 749]], [[671, 746], [689, 770], [727, 739], [710, 719]], [[712, 714], [731, 736], [766, 701], [748, 682]], [[748, 676], [770, 696], [797, 656], [780, 641]]], [[[819, 653], [795, 655], [799, 704], [823, 702]], [[823, 704], [796, 707], [802, 756], [830, 752]]], [[[817, 655], [840, 651], [847, 700], [824, 703]], [[824, 703], [846, 702], [852, 752], [830, 754]]], [[[848, 730], [868, 749], [899, 707], [878, 691]], [[880, 687], [901, 709], [927, 666], [907, 651]]], [[[926, 668], [948, 654], [970, 695], [944, 710]], [[945, 713], [973, 697], [997, 743], [967, 756]], [[969, 761], [995, 745], [1018, 787], [995, 801]]], [[[1163, 634], [1146, 613], [1101, 641], [1119, 662]], [[1115, 666], [1093, 650], [1062, 687], [1084, 705]], [[1080, 710], [1056, 696], [1033, 741], [1056, 754]], [[1054, 754], [1031, 744], [1011, 791], [1032, 800]]], [[[1158, 634], [1173, 651], [1133, 683], [1117, 662]], [[1114, 665], [1133, 682], [1100, 718], [1082, 707]], [[1079, 711], [1098, 722], [1076, 765], [1057, 756]], [[1055, 755], [1076, 766], [1053, 810], [1034, 802]]], [[[1166, 657], [1186, 641], [1218, 678], [1197, 695]], [[1198, 699], [1220, 681], [1251, 718], [1230, 737]], [[1232, 740], [1253, 719], [1285, 757], [1263, 776]], [[1265, 779], [1287, 762], [1316, 797], [1296, 815]], [[1298, 816], [1321, 797], [1349, 841], [1331, 856]]], [[[1295, 650], [1325, 648], [1328, 696], [1300, 700]], [[1300, 701], [1329, 699], [1337, 747], [1306, 750]], [[1308, 755], [1337, 746], [1353, 794], [1325, 803]], [[1328, 808], [1353, 797], [1375, 842], [1350, 852]]], [[[224, 748], [220, 723], [271, 712], [275, 740]], [[277, 741], [277, 712], [331, 717], [325, 744]], [[327, 744], [330, 714], [381, 728], [372, 753]]], [[[394, 769], [406, 744], [451, 767], [439, 792]], [[444, 795], [452, 767], [500, 780], [492, 811]], [[499, 810], [500, 782], [550, 785], [550, 813]]], [[[606, 799], [611, 822], [660, 813], [655, 787]], [[659, 787], [663, 814], [714, 807], [706, 779]], [[709, 776], [714, 806], [765, 799], [756, 769]], [[760, 766], [767, 798], [814, 789], [810, 763]]], [[[826, 798], [844, 781], [881, 816], [860, 832]]], [[[845, 781], [858, 765], [894, 799], [879, 816]]], [[[906, 839], [900, 812], [952, 803], [957, 832]], [[959, 831], [953, 802], [1002, 796], [1008, 825]]], [[[1064, 828], [1052, 805], [1095, 779], [1106, 804]], [[1108, 803], [1098, 776], [1147, 757], [1153, 784]], [[1156, 786], [1155, 758], [1206, 755], [1204, 786]], [[1206, 788], [1218, 757], [1263, 784], [1247, 808]], [[1251, 812], [1269, 787], [1308, 826], [1287, 843]], [[1287, 848], [1308, 827], [1343, 865], [1324, 883]]], [[[220, 767], [237, 741], [276, 770], [260, 794]], [[266, 796], [279, 770], [322, 792], [307, 820]], [[315, 823], [324, 792], [370, 805], [363, 835]], [[368, 837], [374, 804], [418, 811], [417, 844]], [[424, 844], [422, 812], [469, 811], [472, 844]], [[480, 843], [473, 812], [524, 810], [530, 832]]], [[[569, 835], [582, 809], [627, 829], [613, 856]], [[621, 858], [628, 829], [676, 845], [665, 874]], [[675, 875], [675, 844], [726, 851], [721, 880]], [[729, 881], [727, 852], [777, 849], [777, 880]], [[786, 880], [779, 851], [828, 843], [832, 872]], [[837, 868], [827, 841], [875, 828], [886, 855]]]]
"""