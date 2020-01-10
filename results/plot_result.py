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
FID_base = [23.98 for i in id]

total_IS = [(IS_value[i]-4.36)/4.36 for i in range(len(FID_id))]
total_FID = [ - (FID_value[i] - 23.98)/23.98 for i in range(len(FID_id))]
total = [(IS_value[i]-4.36)/4.36 - (FID_value[i] - 23.98)/23.98 for i in range(len(FID_id))]
base = [0 for i in id]

plt.plot(FID_id, FID_value, "o-g", label='FID_test')
plt.plot(FID_id, FID_base, "-r", label='FID_base')
plt.savefig("result-FID.jpg")
plt.close()
plt.errorbar(IS_id, IS_value, yerr=IS_error, ecolor='black', mec='green', ms=20, mew=4)
plt.plot(IS_id, IS_base, "-r", label='IS_base')
plt.savefig("result-IS.jpg")
plt.close()
# plt.plot(IS_id, total_IS, "o-y", label='FID_test')
# plt.plot(FID_id, total_FID, "o-b", label='FID_test')
plt.plot(FID_id, total, "o-g", label='FID_test')
plt.plot(FID_id, base, "-r", label='FID_base')
plt.savefig("result-total.jpg")
with open("result.txt", "a+") as f:
    for i in range(len(total_IS)):
        if total_IS[i] > 0:
            if total_FID[i] > 0:
                f.write("epoch {} ---> IS mean:{}, std:{} / FID: {}\n".format(FID_id[i], IS_value[i], IS_error[i], FID_value[i]))
