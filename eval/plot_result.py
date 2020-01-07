import matplotlib.pyplot as plt
import re

content = open("print-IS.txt", "r").read()
id, value = content.split("\n")[::2], content.split("\n")[1::2]
IS_id = [int(i.split("epoch ")[1]) for i in id]
IS_value = [float(re.findall(r'-?\d+\.?\d*e?-?\d*?', i)[0]) for i in value]
IS_error = [float(re.findall(r'-?\d+\.?\d*e?-?\d*?', i)[1]) for i in value]
IS_base = [4.36 for i in id]
content = open("print-FID.txt", "r").read()
id, value = content.split("\n")[::2], content.split("\n")[1::2]
FID_id = [int(i.split(" ")[1]) for i in id]
FID_value = [float(i.split(":")[1]) for i in value]
FID_base = [22 for i in id]

plt.plot(FID_id, FID_value, "o-g", label='FID_test')
plt.plot(FID_id, FID_base, "-r", label='FID_base')
plt.savefig("result-FID.jpg")
plt.close()
plt.errorbar(IS_id, IS_value, yerr=IS_error, ecolor='black', mec='green', ms=20, mew=4)
plt.plot(IS_id, IS_base, "-r", label='IS_base')
plt.savefig("result-IS.jpg")

