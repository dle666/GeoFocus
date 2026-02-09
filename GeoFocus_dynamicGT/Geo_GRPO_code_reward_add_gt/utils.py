import json
import re
import math
import torch
import torch.distributed as dist
import numpy as np
from logging_setup import logger

with open('Geo_GRPO_reasoning_parser/collinear.json', 'r', encoding='UTF-8') as file:
    collinear = json.load(file)

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getConsCdlAcc(target, prediction):
    '''
    To find the accuracy of a single-sample composition statement, the input form is 'Shape(), Collinear()'
    test>>target="Shape(EDB,BF,EFD),Shape(CD,EFD,FC),Shape(FB,EBF),Collinear(BFC),Cocircular(E,FDB)"
        >>prediction="Shape(EDB,BF,EFD),Shape(CD,EFD,FC),Collinear(BFC),Cocircular(E,FDB)"
    '''
    try:
        target = target.replace(' ', '')
        prediction = prediction.replace(' ', '')

        target = re.split(r',(?![^()]*\))', target)
        prediction = re.split(r',(?![^()]*\))', prediction)

        prediction = list(set(prediction))
        target = list(set(target))

        target_types = []
        target_elems = []
        target_mark = [0] * len(target)
        for i in range(len(target)):
            matches = re.search(r'^(.*?)\((.*?)\)', target[i])

            type = matches.group(1)
            values = matches.group(2).split(',')
            target_types.append(type)
            target_elems.append(values)

        for i in range(len(prediction)):

            matches = re.search(r'^(.*?)\((.*?)\)', prediction[i])

            type = matches.group(1)
            values = matches.group(2).split(',')

            for i in range(len(target)):
                if type == target_types[i] and target_mark[i] == 0:
                    if type == 'Shape' and can_rotate(values, target_elems[i], isList=True):
                        target_mark[i] = 1
                        break
                    elif type == 'Collinear' and (values[0] == target_elems[i][0] or values[0] == target_elems[i][0][::-1]):
                        target_mark[i] = 1
                        break
                    elif type == 'Cocircular':
                        if len(target_elems[i]) == 1 and values[0] == target_elems[i][0]:
                            target_mark[i] = 1
                            break
                        elif values[0] == target_elems[i][0] and can_rotate(values[1],target_elems[i][1],isList=False):
                            target_mark[i] = 1
                            break

        return sum(target_mark) / len(target_mark), len(target)
    except Exception as e:
        logger.info(f"[错误Cons] {e}")
        logger.info(f"[输入target] {target}")
        logger.info(f"[输入prediction] {prediction}")
        return 0, 0

def getImgCdlAcc(proId, target, prediction):
    '''
        To find the accuracy of a single-sample image_cdl, the input form is required to be 'Equal(), ParallelBetweenLine(), PerpendicularBetweenLine'
        The question ID needs to be obtained to resolve the collinear condition
        test>>target="Equal(MeasureOfAngle(HFC),50),ParallelBetweenLine(AE,CF),PerpendicularBetweenLine(CE,BE)"
            >>prediction="Equal(MeasureOfAngle(HFC),50),ParallelBetweenLine(AE,CF)"
    '''
    try:
        target = split_str(target.replace(' ', ''))
        prediction = split_str(prediction.replace(' ', ''))

        if (len(target) == 1 and target[0]=='') or len(target) == 0:
            return 1, 0
        
        if (len(prediction) == 1 and prediction[0]=='') or len(prediction) == 0:
            return 0, len(target)

        prediction = list(set(prediction))
        target = list(set(target))

        target_types = []
        target_elems = []
        target_mark = [0] * len(target)
        for i in range(len(target)):
            matches = re.search(r'^([^(]+)\((.*)\)$', target[i])
            type = matches.group(1)
            values = matches.group(2)

            pattern = r'Add|Sub|Mul|Div|Sin|Cos|Tan|RatioOfSimilarTriangle|RatioOfMirrorSimilarTriangle|RatioOfSimilarQuadrilateral|RatioOfMirrorSimilarQuadrilateral'
            if type == 'Equal' and re.search(pattern, values):
                type == 'Equal-a'
            else:
                values = values.split(',')
            target_types.append(type)
            target_elems.append(values)

        for i in range(len(prediction)):
            matches = re.search(r'^([^(]+)\((.*)\)$', prediction[i])

            type = matches.group(1)
            values = matches.group(2)
            if type == 'Equal' and re.search(r'Add|Sub|Mul|Div|Sin|Cos|Tan', values):
                type == 'Equal-a'

            # Different types of judgment methods are considered in the case of collinearity
            for i in range(len(target)):
                if type == target_types[i] and target_mark[i] == 0:
                    if type == 'Equal':
                        values = values if isinstance(values, list) else values.split(',')
                        p_lineNum = sum('LengthOfLine' in e for e in values)
                        t_lineNum = sum('LengthOfLine' in e for e in target_elems[i])
                        p_angNum = sum('MeasureOfAngle' in e for e in values)
                        t_angNum = sum('MeasureOfAngle' in e for e in target_elems[i])
                        # The target and prediction types are different
                        if (p_lineNum != t_lineNum) or (p_angNum != t_angNum):
                            continue
                        # Judgment of the numerical condition of the line segment length
                        elif p_lineNum == 1 and t_lineNum == 1:
                            p_line = re.split(r'[()]', values[0])[1]
                            t_line = re.split(r'[()]', target_elems[i][0])[1]
                            if (p_line == t_line or p_line[::-1] == t_line) and values[1] == target_elems[i][1]:
                                target_mark[i] = 1
                                break
                        # Judgment of the numerical condition of the angle length
                        elif p_angNum == 1 and t_angNum == 1:
                            p_ang = set(re.split(r'[()]', values[0])[1])
                            t_ang = set(re.split(r'[()]', target_elems[i][0])[1])
                            if values[1] == target_elems[i][1] and p_ang == t_ang:
                                target_mark[i] = 1
                                break
                        # Judgment of equality condition of two line segments
                        elif p_lineNum == 2 and t_lineNum == 2:
                            p_l1, p_l2 = re.split(r'[()]', values[0])[1], re.split(r'[()]', values[1])[1]
                            t_l1, t_l2 = re.split(r'[()]', target_elems[i][0])[1], re.split(r'[()]', target_elems[i][1])[1]
                            set1 = {p_l1, p_l1[::-1], p_l2, p_l2[::-1]}
                            set2 = {t_l1, t_l1[::-1], t_l2, t_l2[::-1]}
                            if set1 == set2:
                                target_mark[i] = 1
                                break
                        # Judgment of equality condition of two angles
                        elif p_angNum == 2 and t_angNum == 2:
                            p_a1 = list(re.split(r'[()]', values[0])[1])
                            p_a2 = list(re.split(r'[()]', values[1])[1])
                            t_a1 = list(re.split(r'[()]', target_elems[i][0])[1])
                            t_a2 = list(re.split(r'[()]', target_elems[i][1])[1])

                            # 判断方式一：p_a1 对应 t_a1，p_a2 对应 t_a2
                            if (
                                p_a1[1] == t_a1[1] and set(p_a1) == set(t_a1) and
                                p_a2[1] == t_a2[1] and set(p_a2) == set(t_a2)
                            ):
                                target_mark[i] = 1
                                break

                            # 判断方式二：p_a1 对应 t_a2，p_a2 对应 t_a1（两角顺序交换）
                            elif (
                                p_a1[1] == t_a2[1] and set(p_a1) == set(t_a2) and
                                p_a2[1] == t_a1[1] and set(p_a2) == set(t_a1)
                            ):
                                target_mark[i] = 1
                                break
                        # Judgment of arc condition
                        else:
                            values = set(values)
                            target_arc = set(target_elems[i])
                            if values == target_arc:
                                target_mark[i] = 1
                                break
                            else:
                                values = list(values)
                    elif type == 'Equal-a' and values == target_elems[i]:
                        target_mark[i] = 1
                        break
                    elif type == 'PerpendicularBetweenLine':
                        values = values if isinstance(values, list) else values.split(',')
                        p1 = set(values[0])
                        p2 = set(values[1])
                        t1 = set(target_elems[i][0][::-1])
                        t2 = set(target_elems[i][1][::-1])
                        if (p1 == t1 and p2 == t2) or (p1 == t2 and p2 == t1):
                            target_mark[i] = 1
                            break
                    elif type == 'ParallelBetweenLine':
                        values = values if isinstance(values, list) else values.split(',')
                        p1 = set(values[0])
                        p2 = set(values[1])
                        t1 = set(target_elems[i][0])
                        t2 = set(target_elems[i][1])
                        if (p1 == t1 and p2 == t2) or (p1 == t2 and p2 == t1):
                            target_mark[i] = 1
                            break

        return sum(target_mark) / len(target_mark), len(target)
    except Exception as e:
        logger.info(f"[错误ImgC] {e}")
        logger.info(f"[输入target] {target}")
        logger.info(f"[输入prediction] {prediction}")
        return 0, 0

def getTextCdlAcc(proId, target,prediction):

    target = split_str(target.replace(' ', ''))
    prediction = split_str(prediction.replace(' ', ''))

    if target == [""]:
        return 1, 0

    prediction = list(set(prediction))

    target_types = []
    target_elems = []
    target_mark = [0] * len(target)
    for i in range(len(target)):
        matches = re.search(r'^([^(]+)\((.*)\)$', target[i])

        type = matches.group(1)
        values = matches.group(2)

        pattern = r'Add|Sub|Mul|Div|Sin|Cos|Tan|RatioOfSimilarTriangle|RatioOfMirrorSimilarTriangle|RatioOfSimilarQuadrilateral|RatioOfMirrorSimilarQuadrilateral'
        if type in ('PerpendicularBetweenLine','ParallelBetweenLine'):
            values = values.split(',')
        elif type == 'Equal' and not re.search(pattern, values):
            type = 'Equal-a'
            values = values.split(',')
        target_types.append(type)
        target_elems.append(values)
    for i in range(len(prediction)):
        matches = re.search(r'^([^(]+)\((.*)\)$', prediction[i])


        type = matches.group(1)
        values = matches.group(2)
        if type == 'Equal' and not re.search(r'Add|Sub|Mul|Div|Sin|Cos|Tan', values):
            type = 'Equal-a'

        # Different types of judgment methods are considered in the case of collinearity
        for i in range(len(target)):
            if type == target_types[i] and target_mark[i] == 0:
                if type == 'Equal-a':
                    values = values if isinstance(values, list) else values.split(',')
                    p_lineNum = sum('LengthOfLine' in e for e in values)
                    t_lineNum = sum('LengthOfLine' in e for e in target_elems[i])
                    p_angNum = sum('MeasureOfAngle' in e for e in values)
                    t_angNum = sum('MeasureOfAngle' in e for e in target_elems[i])
                    # The target and prediction types are different
                    if (p_lineNum != t_lineNum) or (p_angNum != t_angNum):
                        continue
                    # Judgment of the numerical condition of the line segment length
                    elif p_lineNum == 1 and t_lineNum == 1:
                        p_line = re.split(r'[()]', values[0])[1]
                        t_line = re.split(r'[()]', target_elems[i][0])[1]
                        if (p_line == t_line or p_line[::-1] == t_line) and values[1] == target_elems[i][1]:
                            target_mark[i] = 1
                            break
                    # Judgment of the numerical condition of the angle length
                    elif p_angNum == 1 and t_angNum == 1:
                        p_ang = list(re.split(r'[()]', values[0])[1])
                        t_ang = list(re.split(r'[()]', target_elems[i][0])[1])
                        if values[1] == target_elems[i][1] and p_ang[1] == t_ang[1] and \
                                isPCollinear(proId, p_ang[0], t_ang[:2][::-1]) and isPCollinear(proId, p_ang[2],
                                                                                                t_ang[-2:]):
                            target_mark[i] = 1
                            break
                    # Judgment of equality condition of two line segments
                    elif p_lineNum == 2 and t_lineNum == 2:
                        p_l1, p_l2 = re.split(r'[()]', values[0])[1], re.split(r'[()]', values[1])[1]
                        t_l1, t_l2 = re.split(r'[()]', target_elems[i][0])[1], re.split(r'[()]', target_elems[i][1])[1]
                        set1 = {p_l1, p_l1[::-1], p_l2, p_l2[::-1]}
                        set2 = {t_l1, t_l1[::-1], t_l2, t_l2[::-1]}
                        if set1 == set2:
                            target_mark[i] = 1
                            break
                    # Judgment of equality condition of two angles
                    elif p_angNum == 2 and t_angNum == 2:
                        p_a1, p_a2 = list(re.split(r'[()]', values[0])[1]), list(re.split(r'[()]', values[1])[1])
                        t_a1, t_a2 = list(re.split(r'[()]', target_elems[i][0])[1]), list(
                            re.split(r'[()]', target_elems[i][1])[1])
                        if p_a1[1] == t_a1[1] and isPCollinear(proId, p_a1[0], t_a1[:2][::-1]) and isPCollinear(proId,
                                                                                                                p_a1[2],
                                                                                                                t_a1[
                                                                                                                -2:]) and \
                                p_a2[1] == t_a2[1] and isPCollinear(proId, p_a2[0], t_a2[:2][::-1]) and isPCollinear(
                            proId, p_a2[2], t_a2[-2:]):
                            target_mark[i] = 1
                            break
                        elif p_a1[1] == t_a2[1] and isPCollinear(proId, p_a1[0], t_a2[:2][::-1]) and isPCollinear(proId,
                                                                                                                  p_a1[
                                                                                                                      2],
                                                                                                                  t_a2[
                                                                                                                  -2:]) and \
                                p_a2[1] == t_a1[1] and isPCollinear(proId, p_a2[0], t_a1[:2][::-1]) and isPCollinear(
                            proId, p_a2[2], t_a1[-2:]):
                            target_mark[i] = 1
                            break

                    # Judgment of  conditions other than lines and angles
                    else:
                        values = set(values)
                        target_con = set(target_elems[i])
                        if values == target_con:
                            target_mark[i] = 1
                            break
                        else:
                            values = list(values)
                elif type == 'Equal' and values == target_elems[i]:
                    target_mark[i] = 1
                    break
                elif type == 'PerpendicularBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    p1 = list(values[0])
                    p2 = list(values[1])
                    t1 = list(target_elems[i][0][::-1])
                    t2 = list(target_elems[i][1][::-1])
                    if p1[1] == p2[1] and p1[1] == t1[0] and isPCollinear(proId, p1[0], t1) and isPCollinear(proId,
                                                                                                             p2[0], t2):
                        target_mark[i] = 1
                        break
                elif type == 'ParallelBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    p1 = values[0]
                    p2 = values[1]
                    t1 = target_elems[i][0]
                    t2 = target_elems[i][1]
                    if (isLCollinear(proId, p1, t1) and isLCollinear(proId, p2, t2)) or (
                            isLCollinear(proId, p1, t2[::-1]) and isLCollinear(proId, p2, t1[::-1])):
                        target_mark[i] = 1
                        break
                else:
                    if values == target_elems[i]:
                        target_mark[i] = 1
                        break




    return sum(target_mark) / len(target_mark), len(target)

def getGoalCdlAcc(target,prediction):

    target = target.replace(' ', '')
    prediction = prediction.replace(' ', '')

    if target == prediction:
        return 1,1
    else:
        return 0,1

def can_rotate(value1, value2, isList):
    # The length is different and cannot be rotated
    if len(value1) != len(value2):
        return False

    if isList:
        for i in range(len(value1)):
            if value1 == value2:
                return True
            value2 = [value2[-1]] + value2[:-1]
        return False
    else:
        value1 = value1[0]
        value2 = value2[0]
        extended_value1 = value1 + value1

        if value2 in extended_value1:
            return True
        else:
            return False

def isPCollinear(proId, point, t_line):
    '''
    Determine if the point is on a ray in this direction t_line
    such as, point='X',t_line=['N','A']
    The Collinear statement in the composition statement of the joint problem determines whether X is on the ray in the direction of NA
    :return:boolean
    '''
    if point == t_line[1]:
        return True
    clist = collinear[proId]
    if clist is None:
        return False
    for item in clist:
        if t_line[0] in item and t_line[1] in item:
            item = item if item.index(t_line[0]) < item.index(t_line[1]) else item[::-1]
            if point in item and item.index(point) > item.index(t_line[0]):
                return True
    return False


def isLCollinear(proId, line, t_line):
    '''
    Check whether the line is collinearal with the t_line and the direction is the same
    such as, line=['X','Y'],t_line=['N','A']
    Check whether the line is collinearal with the t_line and the direction is the same
    :return:boolean
    '''
    # 重合
    if line == t_line:
        return True
    clist = collinear[proId]
    if clist is None:
        return False
    for item in clist:
        if line[0] in item and line[1] in item and t_line[0] in item and t_line[1] in item:
            if (item.index(line[0]) - item.index(line[1])) * (item.index(t_line[0]) - item.index(t_line[1])) > 0:
                return True
    return False

def split_str(s):
    result = []
    stack = []
    start = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        elif char == ',' and not stack:
            result.append(s[start:i].strip())
            start = i + 1

    result.append(s[start:].strip())
    return result

def save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer,
                    bleu4, cdlAcc, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'cdlAcc': cdlAcc,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        print("This is the best")
        torch.save(state, 'checkpoint'+'/BEST_' + filename)


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def getAccAvg(acc,num):
    for i in range(len(num)):
        try:
            acc[i] = round(acc[i] / num[i], 3)
        except ZeroDivisionError:
            acc[i] = 0
    return acc

def getMSE(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())

    std_devs = []
    for key in common_keys:
        p1 = dict1[key]
        p2 = dict2[key]

        std_dev = np.linalg.norm(np.array(p1) - np.array(p2))
        std_devs.append(std_dev)
    return sum(std_devs)/len(std_devs)
