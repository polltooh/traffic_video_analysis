import file_io
import numpy as np

result_list = file_io.read_file("spain_resdeconv_results/results.txt")
#result_list = file_io.read_file("results/results.txt")
diff_list = list()
for result in result_list:
    img_name, label, infer = result.split(" ")
    diff = abs(float(label) - float(infer))
    #if diff > 3:
    #    count += 1
    #else:
    #    diff_list.append(diff)
    diff_list.append(diff)
    
diff_list = np.array(diff_list)
print(np.mean(diff_list))
print(np.std(diff_list))
