module REGISTER #(parameter DATA_WIDTH = 16) (
		clk,
		rst_n,
		reg_write,
		data_in,
		data_out
	);
  input clk;
  input rst_n;
  input reg_write;
  input  [DATA_WIDTH-1:0] data_in;
  output reg [DATA_WIDTH-1:0] data_out;
	
	always @(posedge clk) begin
		if(!rst_n) begin
			data_out <= {DATA_WIDTH{1'b0}};
		end
		else if(reg_write) begin
			data_out <= data_in;
		end
		else begin
			data_out <= data_out;
		end
	end

endmodule



