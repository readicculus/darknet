import re


class LogAction():
    def __init__(self,pattern,action):
        self.pattern = re.compile(pattern)
        self.action = action

    def try_parse(self, line):
        matches = self.pattern.findall(line)
        if len(matches) < 1:
            return None

        items = matches[0]
        if not self.action is None:
            self.action(items)

        return items

class DarknetLogParser():
    def __init__(self):
        self.iteration_pattern = LogAction(r"([\d].*): .*?, (.*?) avg loss, (.*?) rate, (.*?) seconds, (.*?) images, (.*?) hours left", None)
        self.next_map_pattern = LogAction(r"\(next mAP calculation at (\d*) iterations\)", None)
        self.v3_info_pattern_all = LogAction(r"v3 .*?iou: (.*?), cls: (.*?)\) Region (.*?) Avg \(IOU: (.*?), GIOU: (.*?)\), Class: (.*?), Obj: (.*?), No Obj: (.*?), .5R: (.*?), .75R: (.*?), count: (.*?), class_loss = (.*?), iou_loss = (.*?), total_loss = (.*?)", None)
        self.v3_info_pattern_short = LogAction(r"v3 .*?iou: (.*?),", None)
        self.model_loading_end_pattern = LogAction(r"(Done\! Loaded.*)", None)


        self.precision_by_class_pattern = LogAction(r"class_id\s=\s(\d*), name = (.*), ap =\s([\-\+]?[0-9]*\.[0-9]+)?\%\s*\(TP\s=\s(\d*),\sFP\s=\s(\d*)\)", None)
        self.prf1 = LogAction(r"for conf_thresh = ([\-\+]?[0-9]*\.[0-9]+), precision = ([\-\+]?[0-9]*\.[0-9]+), recall = ([\-\+]?[0-9]*\.[0-9]+), F1-score = ([\-\+]?[0-9]*\.[0-9]+)", None)
        self.tpfpfnioiu = LogAction(r"for conf_thresh = ([\-\+]?[0-9]*\.[0-9]+), TP = (\d*), FP = (\d*), FN = (\d*), average IoU = ([\-\+]?[0-9]*\.[0-9]+) \%", None)
        self.mapatiou = LogAction(r"mean_average_precision\s\(mAP@([\-\+]?[0-9]*\.[0-9]+)\)\s=\s([\-\+]?[0-9]*\.[0-9]+)", None)
        self.saving_weights_pattern = LogAction(r"Saving weights to (.*)", None)
        self.is_model_loading = True
        self.map_by_class = {}
        self.map_scores = []
        self.num_lines_checked = 0

    def parse_map_sections(self, line):
        precision = self.precision_by_class_pattern.try_parse(line)
        if not precision is None:
            classid, name, ap, TP, FP = precision
            if not name in self.map_by_class:
                self.map_by_class[name] = {'ap':[], 'TP':[], 'FP':[]}
            self.map_by_class[name]['ap'].append(ap)
            self.map_by_class[name]['TP'].append(TP)
            self.map_by_class[name]['FP'].append(FP)
            #handle precision parsed and return
            return line

        prf1 = self.prf1.try_parse(line)
        if  not prf1 is None:
            conf_thresh, precision, recall, f1 = prf1
            return line

        tpfpfnioiu = self.tpfpfnioiu.try_parse(line)
        if not tpfpfnioiu is None:
            conf_thresh, TP, FP, FN, avg_IoU = tpfpfnioiu
            return line

        mapatiou = self.mapatiou.try_parse(line)
        if not mapatiou is None:
            IoU, mAP= mapatiou
            return line

        saving_weights = self.saving_weights_pattern.try_parse(line)
        if not saving_weights is None:
            path = saving_weights[0]
            return line
        return None

    def parse(self, line):
        return line
        self.num_lines_checked += 1
        # log the model loading up until model finishes loading
        if self.is_model_loading:
            if self.num_lines_checked > 1000:
                self.is_model_loading = False
            matches = self.model_loading_end_pattern.try_parse(line)
            if not matches is None:
                self.is_model_loading = False
            return line

        matches = self.v3_info_pattern_short.try_parse(line)
        if not matches is None:
            return None

        matches = self.iteration_pattern.try_parse(line)
        if not matches is None:
            return line

        matches = self.next_map_pattern.try_parse(line)
        if not matches is None:
            return line

        return self.parse_map_sections(line)
        # return " ".join(matches)
        return None