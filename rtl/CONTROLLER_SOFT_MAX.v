module CONTROLLER_SOFT_MAX #(parameter DATA_WIDTH = 32, COMPUTE_NUM = 10, IFM_SIZE = 10 , LUT_SIZE = 1) (
		clk,
		rst_n,
		valid_ifm,
		ifm,
		wr_ifm,
		rd_ifm,
		wr_clr,
		rd_clr,
		counter_ifm,
		counter_lut,
		set_output,
		current_state,
		set_reg,
		counter_compute,
		end_softmax
    
	);

  input              clk            ;
  input              rst_n          ;
  input              valid_ifm      ;
  input   [DATA_WIDTH-1:0] ifm      ;
  output wire        wr_ifm         ;
  output reg         rd_ifm         ;
  output reg         wr_clr         ;
  output reg         rd_clr         ;
  output reg         set_output     ;
  output reg [15:0]  counter_ifm    ;
  output reg [3:0]  counter_lut     ;
	output reg [3:0]   current_state  ;
	output reg         set_reg        ;
	output reg         end_softmax    ;
	output reg [3:0]   counter_compute;

	parameter IDLE       = 4'd0;
	parameter WRITE_IFM  = 4'd1;
	parameter WAIT_1     = 4'd2;
	parameter STORE_IFM  = 4'd3;
	parameter WAIT_2     = 4'd4;
	parameter COMPUTE    = 4'd5;
	parameter NOP        = 4'd6;
	parameter CAP_DATA   = 4'd7;
	parameter END        = 4'd8;

	assign wr_ifm = valid_ifm  ;

	reg [3:0 ]  next_state     ;
  
	always @(posedge clk or negedge rst_n) begin
		if(!rst_n) begin
  		current_state <= 0;
  	end
  	else begin
  		current_state <=  next_state;
  	end
	end
  
	always @(valid_ifm or counter_ifm or current_state) begin
		case(current_state)
			IDLE:
				if(valid_ifm & (counter_ifm == 16'd0))
					next_state = WRITE_IFM;
				else 
					next_state = IDLE;
			WRITE_IFM:
				if(counter_ifm == IFM_SIZE)
					next_state = WAIT_1;
				else 
					next_state = WRITE_IFM;
			WAIT_1:
				next_state = current_state + 1;
			WAIT_2:
				next_state = current_state + 1;
			STORE_IFM:
				if(counter_ifm == IFM_SIZE) 
					next_state = WAIT_2;
				else 
					next_state = STORE_IFM;
			COMPUTE:
			  if(counter_ifm == IFM_SIZE)
					next_state = NOP;
				else 
					next_state = COMPUTE;
			NOP:
				if(counter_compute == COMPUTE_NUM + 1) 
					next_state = CAP_DATA;
				else 
					next_state = COMPUTE;
			CAP_DATA:
				next_state = END;
			END:
				next_state = END;
			default: next_state = IDLE;
		endcase
	end
       integer i;
	always @(posedge clk or negedge rst_n) begin
		if(!rst_n) begin
       rd_ifm          <=  0;             			
       wr_clr          <=  1; 
       rd_clr          <=  1; 
       set_output      <=  0; 
       counter_ifm     <=  0; 
			 set_reg         <=  0;
       end_softmax     <=  0;
		end
		else begin
			case(next_state)
				IDLE: begin
          rd_ifm         <=  0;             			
          wr_clr         <=  0; 
          rd_clr         <=  0; 
          set_output     <=  0; 
				end
				WRITE_IFM: begin
          rd_ifm         <=  0;             			
          wr_clr         <=  0; 
          rd_clr         <=  0; 
          set_output     <=  0; 
				end
				STORE_IFM: begin
          rd_ifm         <=  1;             			
          wr_clr         <=  0; 
          rd_clr         <=  0; 
          set_output     <=  0; 
				end
				WAIT_1: begin
          rd_ifm         <=  0;             			
          wr_clr         <=  1; 
          rd_clr         <=  1; 
          set_output     <=  0; 
				end
				WAIT_2: begin
          rd_ifm         <=  0;             			
          wr_clr         <=  1; 
          rd_clr         <=  1; 
          set_output     <=  0; 
				end
				COMPUTE: begin
          rd_ifm         <=  1;             			
          wr_clr         <=  0; 
          rd_clr         <=  0; 
          set_output     <=  0; 
			    set_reg        <=  1;
				end
				CAP_DATA: begin
          rd_ifm         <=  1;             			
          wr_clr         <=  0; 
          rd_clr         <=  0; 
          set_output     <=  0; 
			    set_reg         <=  0;
					end_softmax <= 1;
				end
				NOP: begin
				  rd_clr <= 1;
			    set_reg        <=  0;
          set_output     <=  1; 
				end
				END: end_softmax     <= 1;
			endcase
		end
	end
	
		always @(posedge clk or negedge rst_n) begin
			if(!rst_n) begin
				counter_ifm     <= 16'd0 ;
				counter_lut     <= 4'd0  ;
			  counter_compute <= 4'd0  ;
			end
			else begin
				case(next_state)
					IDLE: begin
				    counter_ifm     <= 16'd0 ;
				    counter_lut     <= 4'd0  ;
			      counter_compute <= 0     ;
				  end
					WRITE_IFM: begin
			 	    counter_ifm <= (wr_ifm) ? (counter_ifm == IFM_SIZE) ? 0 : counter_ifm + 1: counter_ifm;
			 	    counter_lut <= 0;
					end
					WAIT_1: begin
			 	    counter_ifm <= 0;
			 	    counter_lut <= 0;
					end
					STORE_IFM: begin
			 	    counter_ifm <= (rd_ifm) ? (counter_ifm == IFM_SIZE) ? 0 : counter_ifm + 1: counter_ifm;
			 	    counter_lut <= 0;
					end
					COMPUTE: begin
			 	    counter_ifm <= (rd_ifm) ? (counter_ifm == IFM_SIZE) ? 0 : counter_ifm + 1 : counter_ifm;
			 	    counter_lut <= counter_compute + 1;
					end
					NOP: begin
			      counter_compute <=  (counter_compute == COMPUTE_NUM +1) ? 0 : counter_compute +1 ;
			 	    counter_lut <= 0;
					end
					CAP_DATA: begin
			 	    counter_ifm <= 0;
			 	    counter_lut <= 0;
			      counter_compute <=  0;
					end
					default: begin
			 	    counter_ifm <= 0;
			 	    counter_lut <= 0;
					end
				endcase
			end
		end

endmodule

