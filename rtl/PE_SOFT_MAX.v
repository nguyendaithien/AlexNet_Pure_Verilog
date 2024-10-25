module PE_SOFT_MAX #(parameter WGT_WIDTH = 8, DATA_WIDTH = 16, IFM_WIDTH = 8)(
    clk,
    rst_n,
    set_reg, 
    in1,
    in2,
    psum_out
	);

	input clk    ;  
	input rst_n  ;
	input set_reg;
	input  signed [DATA_WIDTH-1:0   ]          in1      ;
	input  signed [DATA_WIDTH-1:0   ]          in2      ;
	output signed [DATA_WIDTH-1:0  ]          psum_out ;
	
	reg signed  [DATA_WIDTH-1:0] psum;
	wire signed [DATA_WIDTH-1:0] psum_in_wire ;
	wire signed [DATA_WIDTH-1:0] psum_out_wire;
  
	REGG #(.DATA_WIDTH(DATA_WIDTH)) psum_reg (
			.clk     (clk          ),
		  .rst_n   (rst_n        ),
		  .set_reg (set_reg      ),
		  .reg_in  (psum         ),
		  .reg_out (psum_out_wire)
	);

	always @(in1 or in2) begin
			psum = in1 + in2;
		end

	assign psum_out = psum_out_wire;
endmodule
