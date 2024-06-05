import torch
# PRA는 Puzzle Reconstruction Accuracy의 약자
# torchmetrics를 모방하여 coding
class PRA():
    def __init__(self):
        self.accuracies = {f"{i}x{i}": 0. for i in range(1, 5)}
        self.counts = {f"{i}x{i}": 0 for i in range(1, 5)}  # 각 크기의 서브 퍼즐에 대한 카운트
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def pra(self, preds, labels):
        preds = torch.argmax(preds, dim=1).view(-1, 16)
        labels = labels.view(-1, 16)
        self.accuracies['1x1'] += (preds == labels).sum().item() / (labels.size(0) * labels.size(1))
        self.counts['1x1'] += 1

        for size in range(2, 5):
            correct_count = 0
            total_subpuzzles = 0
            combinations = [(i, j) for i in range(4 - size + 1) for j in range(4 - size + 1)]

            for predicted_label, label in zip(preds, labels):
                puzzle_a = predicted_label.view(4, 4)
                puzzle_s = label.view(4, 4)

                for start_row, start_col in combinations:
                    rows = slice(start_row, start_row + size)
                    cols = slice(start_col, start_col + size)
                    if torch.equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
                        correct_count += 1
                    total_subpuzzles += 1
            
            self.accuracies[f'{size}x{size}'] += correct_count / total_subpuzzles
            self.counts[f'{size}x{size}'] += 1

    def compute(self):
        averaged_accuracies = {key: self.accuracies[key] / self.counts[key] for key in self.accuracies}
        overall_score = sum(averaged_accuracies.values()) / len(averaged_accuracies)
        return overall_score
    
    def reset(self):
        self.accuracies = {f"{i}x{i}": 0. for i in range(1, 5)}
        self.counts = {f"{i}x{i}": 0 for i in range(1, 5)}

    def get_accuracies(self):
        averaged_accuracies = {key: self.accuracies[key] / self.counts[key] for key in self.accuracies}
        return averaged_accuracies

    def to(self, device):
        self.device = torch.device(device)
    
    def __call__(self, preds, labels):
        self.pra(preds, labels)

    
        
    




# Example usage:
# preds = torch.randint(0, 2, (batch_size, 16), device='cuda')
# labels = torch.randint(0, 2, (batch_size, 16), device='cuda')
# accuracies, score = calc_dacon_metric(preds, labels)
# print(accuracies, score)



######################
### Dacon 제공 코드 ###
######################
# import numpy as np

# def calc_dacon_metric(preds, labels):
#     accuracies = {}
#     accuracies['1x1'] = (preds == labels).sum() / (labels.shape[0] * labels.shape[1])

#     combinations_2x2 = [(i, j) for i in range(3) for j in range(3)]
#     combinations_3x3 = [(i, j) for i in range(2) for j in range(2)]

#     for size in range(2, 5):
#         correct_count = 0  
#         total_subpuzzles = 0

#         for predicted_label, label in zip(preds, labels):
#             puzzle_a = predicted_label.reshape(4, 4)
#             puzzle_s = label.reshape(4, 4)
#             combinations = combinations_2x2 if size == 2 else combinations_3x3 if size == 3 else [(0, 0)]

#             for start_row, start_col in combinations:
#                 rows = slice(start_row, start_row + size)
#                 cols = slice(start_col, start_col + size)
#                 if np.array_equal(puzzle_a[rows, cols], puzzle_s[rows, cols]):
#                     correct_count += 1
#                 total_subpuzzles += 1
#             accuracies[f'{size}x{size}'] = correct_count / total_subpuzzles


#     score = (accuracies['1x1'] + accuracies['2x2'] + accuracies['3x3'] + accuracies['4x4']) / 4.
#     return accuracies