module WRITE_DATA_SOFTMAX #(parameter DATA_WIDTH = 32 , OUTPUT_SIZE = 1 ) (
	   clk
		,rst_n
		,state
		,valid_data
		,sel_data
		,counter_ifm
		,counter_compute
);
  input clk;
  input rst_n;
  input [3:0] state;
  input [3:0] counter_compute;
	input [15:0] counter_ifm;
  output reg valid_data;
  output reg [3:0] sel_data;

	reg [2:0] current_state;
	reg [2:0] next_state;

	parameter IDLE = 3'd0;
	parameter WAIT_WRITE = 3'd1;
	parameter WRITE_DATA = 3'd2;

	always @(posedge clk or negedge rst_n) begin
		if(!rst_n) begin
			current_state <= 3'd0;
		end
		else begin
			current_state <= next_state;
		end
	end

	always @(state or current_state or sel_data or counter_compute or counter_ifm ) begin
		case(current_state)
			3'd0: begin
				if((state == 4'd5) && (counter_ifm == 0) && (counter_compute > 0) )
					next_state = WRITE_DATA;
				else 
					next_state = IDLE;
				end
			3'd2:
				if(sel_data == OUTPUT_SIZE)
					next_state = IDLE;
				else 
					next_state = WRITE_DATA;
			default: next_state = IDLE;
		endcase
	end
	always @(posedge clk or negedge rst_n) begin
		if(!rst_n) begin
			valid_data <= 0;
			sel_data <= 0;
		end
		else begin
			case(next_state) 
				IDLE: begin
					valid_data <= 0;
			    sel_data <= 0;
				end
				WRITE_DATA: begin
					valid_data <= 1;
			    sel_data <= (sel_data == OUTPUT_SIZE + 1) ?  0 : sel_data + 1;
				end
				default: begin
					valid_data <= 0;
			    sel_data <= 0;
				end
			endcase
		end
	end
					
endmodule

					



