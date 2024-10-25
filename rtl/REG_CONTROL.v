module REG_CONTROL #( parameter DATA_WIDTH = 32) (
	current_state,
	counter_ifm,
  reg_write_1,
  reg_write_2,
  reg_write_3,
  reg_write_4,
  reg_write_5,
  reg_write_6,
  reg_write_7,
  reg_write_8,
  reg_write_9,
  reg_write_10,
);

  input [3:0] current_state;
  input [15:0] counter_ifm;
  output reg reg_write_1 ;
  output reg reg_write_2 ;
  output reg reg_write_3 ;
  output reg reg_write_4 ;
  output reg reg_write_5 ;
  output reg reg_write_6 ;
  output reg reg_write_7 ;
  output reg reg_write_8 ;
  output reg reg_write_9 ;
  output reg reg_write_10 ;

	always @(current_state or counter_ifm) begin
        reg_write_1 = 1'b0 ;
        reg_write_2 = 1'b0 ;
        reg_write_3 = 1'b0 ;
        reg_write_4 = 1'b0 ;
        reg_write_5 = 1'b0 ;
        reg_write_6 = 1'b0 ;
        reg_write_7 = 1'b0 ;
        reg_write_8 = 1'b0 ;
        reg_write_9 = 1'b0 ;
        reg_write_10 = 1'b0 ;
		if(current_state == 4'd3) begin
			case(counter_ifm) 
        16'd1  : reg_write_1  = 1'b1 ;
        16'd2  : reg_write_2  = 1'b1 ;
        16'd3  : reg_write_3  = 1'b1 ;
        16'd4  : reg_write_4  = 1'b1 ;
        16'd5  : reg_write_5  = 1'b1 ;
        16'd6  : reg_write_6  = 1'b1 ;
        16'd7  : reg_write_7  = 1'b1 ;
        16'd8  : reg_write_8  = 1'b1 ;
        16'd9  : reg_write_9  = 1'b1 ;
        16'd10 : reg_write_10 = 1'b1 ;
			endcase 
			end
			else begin
        reg_write_1 =  1'b0 ;
        reg_write_2 =  1'b0 ;
        reg_write_3 =  1'b0 ;
        reg_write_4 =  1'b0 ;
        reg_write_5 =  1'b0 ;
        reg_write_6 =  1'b0 ;
        reg_write_7 =  1'b0 ;
        reg_write_8 =  1'b0 ;
        reg_write_9 =  1'b0 ;
        reg_write_10 = 1'b0 ;
			end
		end

endmodule
