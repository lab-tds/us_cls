from sklearn.ensemble import RandomForestClassifier
from functions import BasicParams

# estimators = []
# model = RandomForestClassifier(n_estimators=100, max_depth=7)
# cls = 'rf'
# for i in range(10):
#     estimators.append((f'rf{i}', model))

# print(estimators)

bp = BasicParams()
#bp.getBestParams('./')
bp.defineBestParams('./')