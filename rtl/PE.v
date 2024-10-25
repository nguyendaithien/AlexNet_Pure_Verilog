module PE #(parameter WEIGHT_WIDTH = 8, IFM_WIDTH = 8, PSUM_WIDTH = 16, POOLING = 0)(
    clk,
    rst_n,
    set_reg,
		ifm,
		wgt,
		psum_in,
		psum_out
		);

	input clk    ;  
	input rst_n  ;
	input set_reg;
	input  signed [IFM_WIDTH-1:0   ] ifm      ;
	input  signed [WEIGHT_WIDTH-1:0] wgt      ;
	input  signed [PSUM_WIDTH-1:0  ] psum_in  ;
	output signed [PSUM_WIDTH-1:0  ] psum_out ;
	
	reg  signed [PSUM_WIDTH-1:0] psum         ;
	wire signed	[PSUM_WIDTH-1:0] product			;

	qmult #(
		 .Q_a(12)
		,.N_a(IFM_WIDTH)
		,.Q_b(12)
		,.N_b(WEIGHT_WIDTH)
		,.Q_q(12)
		,.N_q(PSUM_WIDTH)	
	) mult
	(
		 .a(ifm)
		,.b(wgt)
		,.q_result(product)
	);

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
      psum <= 0;
    else 
    begin
      if (set_reg)
      begin
        if (POOLING)
          psum <= (ifm > psum_in) ? ifm : psum_in;
        else
          psum <= product + psum_in;
      end
      else
        psum <= psum;
    end
  end

	assign psum_out = psum;
endmodule
