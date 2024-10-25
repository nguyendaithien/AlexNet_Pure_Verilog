module SUBTRACT #(parameter DATA_WIDTH = 32) (
		op_a,
		op_b,
		result
	);
	input  signed  [DATA_WIDTH-1:0] op_a;
	input  signed  [DATA_WIDTH-1:0] op_b;
	output signed [DATA_WIDTH-1:0] result;
	//wire [DATA_WIDTH-1:0] op_a_2;
	//assign op_a_2 = ~(op_a) + 1;

	//assign result = op_b + op_a_2;
  assign result = op_b - op_a;

endmodule

