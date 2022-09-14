import math

def sumToNM(value1, value2):
    result = 0
    for n in range(value1, value2 + 1):
        result += n
    return result

print(sumToNM(5,25))
print(sumToNM(100,512))