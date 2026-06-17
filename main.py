from uncertainty import Uncertain

x = Uncertain(1, 0.5)
y = Uncertain(1, 0.5)

print(x + y)

x.add_corr(y, 1)
print(x - y)
