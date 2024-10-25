module qmult #(
  // Parameterized values
  parameter Q_a = 8,
  parameter N_a = 16,
  parameter Q_b = 10,
  parameter N_b = 16,
  parameter Q_q = 12,
  parameter N_q = 16
)
(
  input      [N_a-1:0] a,
  input      [N_b-1:0] b,
  output     [N_q-1:0] q_result  // Output quantized to the same number of bits as the input
);

  wire [N_a+N_b-1:0] f_result;  // Multiplication by 2 values of N bits requires a register that is N+N = 2N deep
  wire [N_a-1:0] multiplicand;  // Representing input a
  wire [N_b-1:0] multiplier;    // Representing input b
  wire [N_q-2:0] quantized_result;

  assign multiplicand = (a[N_a-1]) ? (~a) + 1 : a;
  assign multiplier = (b[N_b-1]) ? (~b) + 1 : b;

  assign f_result = multiplicand[N_a-2:0] * multiplier[N_b-2:0];         // Remove sign bit for multiplication
  assign quantized_result = f_result[N_q-2+Q_q:Q_q];                     // Quantize output to the required number of bits
  assign q_result[N_q-1] = (a != 0 && b != 0 && quantized_result != 0) ? a[N_a-1] ^ b[N_b-1] : 0;  // Sign bit of output is XOR of input sign bits
  assign q_result[N_q-2:0] = (q_result[N_q-1]) ? (~quantized_result) + 1 : quantized_result;  // Return 2's complement representation if the result is negative

endmodule
