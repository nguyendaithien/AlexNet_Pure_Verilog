module MUX_10_TO_1 #(parameter DATA_WIDTH = 24) (
		 in0
		,in1
		,in2
		,in3
		,in4
		,in5
		,in6
		,in7
		,in8
		,in9
		,sel
		,out
);
  input  [DATA_WIDTH-1:0] in0; 
  input  [DATA_WIDTH-1:0] in1;
  input  [DATA_WIDTH-1:0] in2;
  input  [DATA_WIDTH-1:0] in3;
  input  [DATA_WIDTH-1:0] in4;
  input  [DATA_WIDTH-1:0] in5;
  input  [DATA_WIDTH-1:0] in6;
  input  [DATA_WIDTH-1:0] in7;
  input  [DATA_WIDTH-1:0] in8;
  input  [DATA_WIDTH-1:0] in9;
  input  [3:0] sel;
  output reg [DATA_WIDTH-1:0] out;

	always @(*) begin
		case(sel)
			4'd1 : out = in0;
			4'd2 : out = in1;
			4'd3 : out = in2;
			4'd4 : out = in3;
			4'd5 : out = in4;
			4'd6 : out = in5;
			4'd7 : out = in6;
			4'd8 : out = in7;
			4'd9 : out = in8;
			4'd10: out = in9;
	  	default : out = 0;
		endcase
	end

endmodule
			



