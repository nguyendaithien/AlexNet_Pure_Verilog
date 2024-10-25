import math

def fixed_point_to_float(binary_representation, q_bits, i_bits):
  # Get the sign bit
  sign_bit = binary_representation[0]

  # Convert the binary value to an integer
  binary_value = binary_representation[1:]
  if sign_bit == '1':
    binary_value = twos_complement(binary_value, q_bits + i_bits - 1)
    value = -int(binary_value, 2)
  else:
    value = int(binary_value, 2)

  # Convert the integer value back to a float
  float_value = value / pow(2, q_bits)                                                                
  
  return float_value

def twos_complement(binary_value, num_bits):
  # Determine the complement value
  complement_value = ''.join('1' if bit == '0' else '0' for bit in binary_value)
  
  # Add leading zeros if necessary
  if len(complement_value) < num_bits:
    complement_value = '1' + complement_value.zfill(num_bits - 1)
  # Add 1 to the complement value
  twos_complement_value = bin(int(complement_value, 2) + 1)[2:].zfill(num_bits)

  return twos_complement_value

def convert_file_values_max(file_path, q_bits, i_bits):
  max_value = float('-inf')
  index = 0
  max_position = None
  with open(file_path, 'r') as file:
    for line in file:
      binary_rep = line.strip()  # Remove leading/trailing whitespaces and newline characters
      float_val = fixed_point_to_float(binary_rep, q_bits, i_bits)
      if float_val > max_value:
        max_value = float_val
        max_position = index
      index += 1
  
  return max_value, max_position

def convert_file_values_min(file_path, q_bits, i_bits):
  min_value = float('inf')
  index = 0
  min_position = None
  with open(file_path, 'r') as file:
    for line in file:
      binary_rep = line.strip()  # Remove leading/trailing whitespaces and newline characters
      float_val = fixed_point_to_float(binary_rep, q_bits, i_bits)
      if float_val < min_value:
        min_value = float_val
        min_position = index
      index += 1
  
  return min_value, min_position

label_path = 'label.txt'
python_path = 'ofm.txt'
file_path = 'ofm_rtl_1.txt'  # Replace with the path to your file
q_bits = 12  # Replace with the number of quantization bits
i_bits = 10  # Replace with the number of integer bits

max_value, max_position = convert_file_values_max(python_path, q_bits, i_bits)
value, position = convert_file_values_max(file_path, q_bits, i_bits)
with open(label_path, 'r') as file:
  label = int(file.readline().strip())
print(f"Label: {label}")
print(f"Detect in python: {max_position}")
print(f"Detect in RTL   : {position}")
if (label == position and label == max_position):
  print("-->Correct.\n")
else:
  print("-->Fail.\n")
