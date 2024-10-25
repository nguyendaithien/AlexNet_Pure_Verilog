module SOFTMAX_TOP #(parameter DATA_WIDTH_IN = 32, NUM_SUBTRACT = 10,  IFM_SIZE = 10, LUT_SIZE = 100, DATA_WIDTH_OUT = 32, NUM_REG_IFM = 10, NUM_COMPUTE = 10)  (
		clk1              ,
		clk2              ,
		rst_n             ,
		valid_ifm         ,
		ifm               ,
		softmax_out_final ,
		valid_data        ,
		ifm_read          ,
    end_softmax
);
  input clk1                                          ;
  input clk2                                          ;
  input rst_n                                         ;
  input valid_ifm                                     ;
  input  [DATA_WIDTH_IN -1:0 ] ifm                    ;
  output reg [DATA_WIDTH_OUT- 1:0] softmax_out_final  ;
  output wire valid_data                              ;
	output wire ifm_read                                ;
  output wire end_softmax                             ;

	assign ifm_read = valid_ifm                 ;
	wire wr_ifm                                 ;
	wire rd_ifm                                 ;
	wire wr_clr                                 ;
	wire rd_clr                                 ;
	wire [0:0] reg_write [9:0]                  ;
	wire reg_write_output                       ;
	wire set_output                             ;
	wire [DATA_WIDTH_IN-1:0] data_out_reg [9:0] ;

	wire [DATA_WIDTH_IN-1:0] data_out_fifo   ;
	wire [15:0] counter_ifm                  ;
	wire [3:0] counter_lut                   ;

  wire [ DATA_WIDTH_IN-1:0] result [ 9:0]  ;
	
	wire [3:0] sel_mux_output;
	wire [DATA_WIDTH_IN-1:0] data_out_mux;
	wire flag_w;
	assign flag_w = ((counter_ifm == 0) || (counter_ifm == 1));
	wire set_reg;


wire [DATA_WIDTH_OUT-1:0] data_out_lut;

reg [DATA_WIDTH_OUT-1:0] reg_out  ;

wire [3:0] current_state;
wire [3:0] counter_compute;

CONTROLLER_SOFT_MAX #(.DATA_WIDTH(DATA_WIDTH_IN), .IFM_SIZE(IFM_SIZE), .LUT_SIZE(IFM_SIZE),.COMPUTE_NUM(NUM_COMPUTE)) controller (
		 .clk            ( clk1             )
		,.rst_n          ( rst_n            )
		,.valid_ifm      ( valid_ifm        )
		,.ifm            ( ifm              )
		,.wr_ifm         ( wr_ifm           )
		,.rd_ifm         ( rd_ifm           )
		,.wr_clr         ( wr_clr           )
		,.rd_clr         ( rd_clr           )
		,.set_output     ( reg_write_output )
		,.counter_ifm    ( counter_ifm      )
		,.counter_lut    ( counter_lut      )
		,.current_state  ( current_state    )
		,.set_reg        ( set_reg          )
		,.counter_compute( counter_compute  )
    ,.end_softmax    ( end_softmax)
	);

FIFO_ASYNCH_SOFTMAX #(.DATA_WIDTH(DATA_WIDTH_IN), .FIFO_SIZE(IFM_SIZE), .ADD_WIDTH(10)) ifm_buffer (
		 .clk1          ( clk1          )
		,.clk2          ( clk1          )
		,.rd_clr        ( rd_clr        )
		,.wr_clr        ( wr_clr        )
		,.rd_inc        ( 1'b1          )
		,.wr_inc        ( 1'b1          )
		,.wr_en         ( wr_ifm        )
		,.rd_en         ( rd_ifm        )
		,.data_in_fifo  ( ifm           )
		,.data_out_fifo ( data_out_fifo )
		);

WRITE_DATA_SOFTMAX #(.DATA_WIDTH(DATA_WIDTH_OUT), .OUTPUT_SIZE(1)) write_data (
	   .clk(clk1)
		,.rst_n(rst_n)
		,.state(current_state)
		,.valid_data(valid_data)
		,.sel_data(sel_mux_output)
		,.counter_compute(counter_compute)
		,.counter_ifm(counter_ifm)
);
//=======================================================================================
//=======================================================================================
 genvar i;
   generate
     for (i = 0; i < NUM_REG_IFM ; i = i + 1) begin : REGISTER_INST
       REGISTER #(DATA_WIDTH_IN) reg_inst (
          .clk       ( clk1            )
         ,.rst_n     ( rst_n           )
         ,.reg_write ( reg_write[i]    )
         ,.data_in   ( data_out_fifo   )
         ,.data_out  ( data_out_reg[i] )
       );
     end
   endgenerate

//=======================================================================================
//----------------------------SUBTRACT----------------------------------------------
//=======================================================================================
	 genvar j;
 generate
     for (j = 0; j < NUM_SUBTRACT; j = j + 1) begin : SUBTRACT_INST
         SUBTRACT #(DATA_WIDTH_IN) subtract_inst (
             .op_a   ( data_out_reg[j] ) ,
             .op_b   ( data_out_fifo   ) ,
             .result ( result[j]       )
         );
     end
 endgenerate

 wire [ DATA_WIDTH_OUT-1:0] psum_out;

 wire [ DATA_WIDTH_OUT-1:0] psum_in;

 assign psum_in  = (flag_w) ? 0 : psum_out ;

 
//=======================================================================================
//----------------------------REGISTER TO LUT----------------------------------------------
//=======================================================================================
MUX_10_TO_1 #(.DATA_WIDTH(DATA_WIDTH_IN)) mux_to_lut (
		 .in0 ( result[0]   )
		,.in1 ( result[1]   )
		,.in2 ( result[2]   )
		,.in3 ( result[3]   )
		,.in4 ( result[4]   )
		,.in5 ( result[5]   )
		,.in6 ( result[6]   )
		,.in7 ( result[7]   )
		,.in8 ( result[8]   )
		,.in9 ( result[9]   )
		,.sel ( counter_lut )
		,.out ( data_out_mux)
);
//=======================================================================================
//----------------------------LUT----------------------------------------------
//=======================================================================================

LUT #(.DATA_WIDTH(DATA_WIDTH_IN)) lut(
		.data_in(data_out_mux)
	 ,.data_out(data_out_lut)
);
//=======================================================================================
//----------------------------ADDER ----------------------------------------------
//=======================================================================================
PE_SOFT_MAX #(.DATA_WIDTH(DATA_WIDTH_OUT)) pe1 (
     .clk      ( clk1         )
    ,.rst_n    ( rst_n        )
    ,.set_reg  ( set_reg      )
    ,.in1      ( psum_in      )
    ,.in2      ( data_out_lut )
    ,.psum_out ( psum_out     )
	);


always @(posedge clk1 or negedge rst_n) begin
	if(!rst_n) begin
     softmax_out_final  <= 24'd0;   
	end
	else begin
		softmax_out_final  <= (reg_write_output) ? psum_out  :  softmax_out_final ;  
	end
end

REG_CONTROL #(.DATA_WIDTH(DATA_WIDTH_IN)) reg_control (
		 .current_state ( current_state )
		,.counter_ifm   ( counter_ifm   )
    ,.reg_write_1   ( reg_write[0]  )
    ,.reg_write_2   ( reg_write[1]  )
    ,.reg_write_3   ( reg_write[2]  )
    ,.reg_write_4   ( reg_write[3]  )
    ,.reg_write_5   ( reg_write[4]  )
    ,.reg_write_6   ( reg_write[5]  )
    ,.reg_write_7   ( reg_write[6]  )
    ,.reg_write_8   ( reg_write[7]  )
    ,.reg_write_9   ( reg_write[8]  )
    ,.reg_write_10  ( reg_write[9]  )
	);

endmodule
