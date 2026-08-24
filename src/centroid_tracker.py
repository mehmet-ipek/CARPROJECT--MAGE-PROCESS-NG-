import math
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        self.next_object_id = 1

        # Veri sırasını ve tutarlılığını korumak için OrderedDict kullanıyoruz
        self.objects = OrderedDict()      # object_id -> (center_x, center_y)
        self.boxes = OrderedDict()        # object_id -> (x, y, w, h)
        self.disappeared = OrderedDict()  # object_id -> disappeared_frame_count

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, rect):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.boxes:
            del self.boxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        # 1. Hiç tespit yoksa kadrajdaki tüm nesnelerin kaybolma sayacını artır
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 2. Yeni tespitlerin merkez noktalarını (centroids) hesapla
        input_centroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids.append((cX, cY))

        # 3. Eğer şu an takip edilen hiçbir nesne yoksa hepsini kaydet
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects[i])
            return self.objects

        # 4. Global Mesafe Matrisi (Pairwise Distance Sorting)
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        distances = []
        for r, obj_centroid in enumerate(object_centroids):
            for c, inp_centroid in enumerate(input_centroids):
                dist = math.hypot(
                    obj_centroid[0] - inp_centroid[0],
                    obj_centroid[1] - inp_centroid[1]
                )
                distances.append((dist, r, c))

        # Mesafelere göre küçükten büyüğe sırala (En yakın olanlar öncelikli eşleşir)
        distances.sort(key=lambda x: x[0])

        used_rows = set()
        used_cols = set()

        for dist, r, c in distances:
            if r in used_rows or c in used_cols:
                continue

            # Maksimum eşleşme mesafesi kontrolü
            if dist > self.max_distance:
                continue

            object_id = object_ids[r]
            self.objects[object_id] = input_centroids[c]
            self.boxes[object_id] = rects[c]
            self.disappeared[object_id] = 0

            used_rows.add(r)
            used_cols.add(c)

        # 5. Eşleşemeyen eski nesnelerin kaybolma sayacını güncelle
        unused_rows = set(range(len(object_centroids))).difference(used_rows)
        for r in unused_rows:
            object_id = object_ids[r]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # 6. Eşleşemeyen yeni tespitleri sisteme yeni ID olarak kaydet
        unused_cols = set(range(len(input_centroids))).difference(used_cols)
        for c in unused_cols:
            self.register(input_centroids[c], rects[c])

        return self.objects

    def get_boxes(self):
        return self.boxes

    def get_centroid(self, object_id):
        return self.objects.get(object_id, None)

    def get_box(self, object_id):
        return self.boxes.get(object_id, None)