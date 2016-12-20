import file_io
import numpy as np
import matplotlib.pyplot as plt

result_list = file_io.read_file("resdeconv_results/results.txt")
result_list.sort()
#result_list = file_io.read_file("results/results.txt")
diff_list = list()
true_count = list()
estimate_count = list()
for result in result_list:
    img_name, label, infer = result.split(" ")
    diff = abs(float(label) - float(infer))
    diff_list.append(diff)
    true_count.append(float(label))
    estimate_count.append(float(infer))
    
diff_list = np.array(diff_list)
print(np.mean(diff_list))
print(np.mean(np.square(diff_list)))

plt.plot(true_count, 'g')
plt.plot(estimate_count, 'r')
plt.show()
