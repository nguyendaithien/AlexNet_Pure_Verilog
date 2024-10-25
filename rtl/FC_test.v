module FC #(parameter DATA_WIDTH = 16, IFM_WIDTH = 8, WGT_WIDTH = 8, IFM_SIZE = 9216, KERNEL_SIZE = 4096, TILING_SIZE = 8, RELU = 1) (
	 clk1
	,clk2
	,rst_n
	,ifm          // input fM
	,valid_ifm  // signal when input valid
	,wgt           // weight
	,ofm                
	,valid_data
	,ifm_read
	,wgt_read
);
input  clk1;
input  clk2;
input  rst_n;
input  valid_ifm;
input  [IFM_WIDTH-1:0] ifm;
input  [TILING_SIZE*WGT_WIDTH-1:0] wgt;
output [DATA_WIDTH-1:0] ofm; 	
output  ifm_read;
output  wgt_read;
output  valid_data;

localparam ADD_WIDTH = $clog2(IFM_SIZE)+1;

wire   [IFM_WIDTH-1:0] ifm_out;

wire [DATA_WIDTH-1:0] psum_out [TILING_SIZE-1:0];

wire [31:0] counter_ifm;
wire [31:0] counter_tiling;

wire wr_en;
wire wr_buff_ifm;
wire rd_buff_ifm;
wire set_reg;
wire wr_ifm_clr;
wire rd_ifm_clr;

wire last_channel;
wire end_compute;

wire flag;
assign flag = ((counter_ifm == 0));

reg [DATA_WIDTH - 1:0] regg [TILING_SIZE-1:0];

wire [3:0] sel_mux;
wire [DATA_WIDTH-1:0] out_mux;
wire set_output;
wire [2:0] current_state;

genvar i;
generate
  for (i = 0; i < TILING_SIZE; i = i + 1)
  begin
    always @(posedge clk2 or negedge rst_n) 
    begin
    	if(!rst_n)
      begin
        regg[i] <= 0;
      end
    	else
      begin
    	  regg[i] <= (set_output) ? psum_out[i] : regg[i];
      end
    end
  end
endgenerate

assign out_mux = regg[sel_mux-1];
assign ofm = (RELU)? ((out_mux[DATA_WIDTH-1]) ? 0 : out_mux) : out_mux;

WRITE_DATA #(.DATA_WIDTH(DATA_WIDTH), .TILING_SIZE(TILING_SIZE))  write_data(
		.clk(clk2)
	 ,.rst_n(rst_n)
	 ,.counter_tiling(counter_tiling)
	 ,.state(current_state)
	 ,.data_output()
	 ,.valid_data(valid_data)
	 ,.sel_data(sel_mux)
	);

FC_CONTROL #(.IFM_SIZE(IFM_SIZE), .TILING_SIZE(TILING_SIZE), .KERNEL_SIZE(KERNEL_SIZE))  controller (
   .clk1(clk1)
	,.clk2(clk2)
	,.rst_n(rst_n)
	,.start(1'b0)
	,.ifm_read(ifm_read)
	,.wgt_read(wgt_read)
	,.valid_ifm(valid_ifm)
	,.last_kernel(last_kernel)
	,.end_compute(end_compute)
	,.wr_buff_ifm(wr_buff_ifm)
	,.rd_buff_ifm(rd_buff_ifm)
	,.set_reg(set_reg)
	,.wr_ifm_clr(wr_ifm_clr)
	,.rd_ifm_clr(rd_ifm_clr)
	,.counter_ifm(counter_ifm)
	,.set_output(set_output)
	,.current_state(current_state)
	,.counter_tiling(counter_tiling)
);


FIFO_ASYNCH #(.DATA_WIDTH(DATA_WIDTH), .FIFO_SIZE(IFM_SIZE), .ADD_WIDTH(ADD_WIDTH)) ifm_buffer(
	.clk1(clk1)  ,
  .clk2(clk2)  ,
  .rd_clr(rd_ifm_clr),
  .wr_clr(wr_ifm_clr),
  .rd_inc(1'b1),
  .wr_inc(1'b1),
  .wr_en(wr_buff_ifm) ,
  .rd_en(rd_buff_ifm) ,
  .data_in_fifo(ifm) ,
  .data_out_fifo(ifm_out)
	);

genvar arr_i;
generate
  for (arr_i = 0; arr_i < TILING_SIZE; arr_i = arr_i + 1)
  begin
    PE #(.PSUM_WIDTH(DATA_WIDTH), .WEIGHT_WIDTH(WGT_WIDTH), .IFM_WIDTH(DATA_WIDTH), .POOLING(0)) pe (
    		.clk(clk1)
    	 ,.rst_n(rst_n)
    	 ,.set_reg(set_reg)
    	 ,.ifm(ifm_out)
    	 ,.wgt(wgt[(arr_i+1)*WGT_WIDTH-1:arr_i*WGT_WIDTH])
    	 ,.psum_in((flag)?{DATA_WIDTH{1'b0}}:psum_out[arr_i])
    	 ,.psum_out(psum_out[arr_i])
    	 );
  end
endgenerate

endmodule
