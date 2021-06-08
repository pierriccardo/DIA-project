
"""
classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]

f = ["Y", "A"]

classes_1 = [c for c in classes if f[0] in c]
classes_2 = [c for c in classes if f[1] in c]

print(classes_1)
print(classes_2)
"""

classes = [["Y", "I"], ["Y", "D"], ["A", "I"], ["A", "D"]]
prova = ["Y", "I"]

print(prova in classes)