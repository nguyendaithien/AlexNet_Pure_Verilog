module MUX_100_TO_1 #( parameter DATA_WIDTH = 32) (
		  sel,
			out,
      in_1,      
      in_2, 
      in_3, 
      in_4, 
      in_5, 
      in_6, 
      in_7, 
      in_8, 
      in_9, 
      in_10, 
      in_11, 
      in_12, 
      in_13, 
      in_14, 
      in_15, 
      in_16, 
      in_17, 
      in_18, 
      in_19, 
      in_20, 
      in_21, 
      in_22, 
      in_23, 
      in_24, 
      in_25, 
      in_26, 
      in_27, 
      in_28, 
      in_29, 
      in_30, 
      in_31, 
      in_32, 
      in_33, 
      in_34, 
      in_35, 
      in_36, 
      in_37, 
      in_38, 
      in_39, 
      in_40, 
      in_41, 
      in_42, 
      in_43, 
      in_44, 
      in_45, 
      in_46, 
      in_47, 
      in_48, 
      in_49, 
      in_50, 
      in_51, 
      in_52, 
      in_53, 
      in_54, 
      in_55, 
      in_56, 
      in_57, 
      in_58, 
      in_59, 
      in_60, 
      in_61, 
      in_62, 
      in_63, 
      in_64, 
      in_65, 
      in_66, 
      in_67, 
      in_68, 
      in_69, 
      in_70, 
      in_71, 
      in_72, 
      in_73, 
      in_74, 
      in_75, 
      in_76, 
      in_77, 
      in_78, 
      in_79, 
      in_80, 
      in_81, 
      in_82, 
      in_83, 
      in_84, 
      in_85, 
      in_86, 
      in_87, 
      in_88, 
      in_89, 
      in_90, 
      in_91, 
      in_92, 
      in_93, 
      in_94, 
      in_95, 
      in_96, 
      in_97, 
      in_98, 
      in_99, 
      in_100

);
    input wire [DATA_WIDTH-1:0] in_1; 
    input wire [DATA_WIDTH-1:0] in_2; 
    input wire [DATA_WIDTH-1:0] in_3; 
    input wire [DATA_WIDTH-1:0] in_4; 
    input wire [DATA_WIDTH-1:0] in_5; 
    input wire [DATA_WIDTH-1:0] in_6; 
    input wire [DATA_WIDTH-1:0] in_7; 
    input wire [DATA_WIDTH-1:0] in_8; 
    input wire [DATA_WIDTH-1:0] in_9; 
    input wire [DATA_WIDTH-1:0] in_10; 
    input wire [DATA_WIDTH-1:0] in_11; 
    input wire [DATA_WIDTH-1:0] in_12; 
    input wire [DATA_WIDTH-1:0] in_13; 
    input wire [DATA_WIDTH-1:0] in_14; 
    input wire [DATA_WIDTH-1:0] in_15; 
    input wire [DATA_WIDTH-1:0] in_16; 
    input wire [DATA_WIDTH-1:0] in_17; 
    input wire [DATA_WIDTH-1:0] in_18; 
    input wire [DATA_WIDTH-1:0] in_19; 
    input wire [DATA_WIDTH-1:0] in_20; 
    input wire [DATA_WIDTH-1:0] in_21; 
    input wire [DATA_WIDTH-1:0] in_22; 
    input wire [DATA_WIDTH-1:0] in_23; 
    input wire [DATA_WIDTH-1:0] in_24; 
    input wire [DATA_WIDTH-1:0] in_25; 
    input wire [DATA_WIDTH-1:0] in_26; 
    input wire [DATA_WIDTH-1:0] in_27; 
    input wire [DATA_WIDTH-1:0] in_28; 
    input wire [DATA_WIDTH-1:0] in_29; 
    input wire [DATA_WIDTH-1:0] in_30; 
    input wire [DATA_WIDTH-1:0] in_31; 
    input wire [DATA_WIDTH-1:0] in_32; 
    input wire [DATA_WIDTH-1:0] in_33; 
    input wire [DATA_WIDTH-1:0] in_34; 
    input wire [DATA_WIDTH-1:0] in_35; 
    input wire [DATA_WIDTH-1:0] in_36; 
    input wire [DATA_WIDTH-1:0] in_37; 
    input wire [DATA_WIDTH-1:0] in_38; 
    input wire [DATA_WIDTH-1:0] in_39; 
    input wire [DATA_WIDTH-1:0] in_40; 
    input wire [DATA_WIDTH-1:0] in_41; 
    input wire [DATA_WIDTH-1:0] in_42; 
    input wire [DATA_WIDTH-1:0] in_43; 
    input wire [DATA_WIDTH-1:0] in_44; 
    input wire [DATA_WIDTH-1:0] in_45; 
    input wire [DATA_WIDTH-1:0] in_46; 
    input wire [DATA_WIDTH-1:0] in_47; 
    input wire [DATA_WIDTH-1:0] in_48; 
    input wire [DATA_WIDTH-1:0] in_49; 
    input wire [DATA_WIDTH-1:0] in_50; 
    input wire [DATA_WIDTH-1:0] in_51; 
    input wire [DATA_WIDTH-1:0] in_52; 
    input wire [DATA_WIDTH-1:0] in_53; 
    input wire [DATA_WIDTH-1:0] in_54; 
    input wire [DATA_WIDTH-1:0] in_55; 
    input wire [DATA_WIDTH-1:0] in_56; 
    input wire [DATA_WIDTH-1:0] in_57; 
    input wire [DATA_WIDTH-1:0] in_58; 
    input wire [DATA_WIDTH-1:0] in_59; 
    input wire [DATA_WIDTH-1:0] in_60; 
    input wire [DATA_WIDTH-1:0] in_61; 
    input wire [DATA_WIDTH-1:0] in_62; 
    input wire [DATA_WIDTH-1:0] in_63; 
    input wire [DATA_WIDTH-1:0] in_64; 
    input wire [DATA_WIDTH-1:0] in_65; 
    input wire [DATA_WIDTH-1:0] in_66; 
    input wire [DATA_WIDTH-1:0] in_67; 
    input wire [DATA_WIDTH-1:0] in_68; 
    input wire [DATA_WIDTH-1:0] in_69; 
    input wire [DATA_WIDTH-1:0] in_70; 
    input wire [DATA_WIDTH-1:0] in_71; 
    input wire [DATA_WIDTH-1:0] in_72; 
    input wire [DATA_WIDTH-1:0] in_73; 
    input wire [DATA_WIDTH-1:0] in_74; 
    input wire [DATA_WIDTH-1:0] in_75; 
    input wire [DATA_WIDTH-1:0] in_76; 
    input wire [DATA_WIDTH-1:0] in_77; 
    input wire [DATA_WIDTH-1:0] in_78; 
    input wire [DATA_WIDTH-1:0] in_79; 
    input wire [DATA_WIDTH-1:0] in_80; 
    input wire [DATA_WIDTH-1:0] in_81; 
    input wire [DATA_WIDTH-1:0] in_82; 
    input wire [DATA_WIDTH-1:0] in_83; 
    input wire [DATA_WIDTH-1:0] in_84; 
    input wire [DATA_WIDTH-1:0] in_85; 
    input wire [DATA_WIDTH-1:0] in_86; 
    input wire [DATA_WIDTH-1:0] in_87; 
    input wire [DATA_WIDTH-1:0] in_88; 
    input wire [DATA_WIDTH-1:0] in_89; 
    input wire [DATA_WIDTH-1:0] in_90; 
    input wire [DATA_WIDTH-1:0] in_91; 
    input wire [DATA_WIDTH-1:0] in_92; 
    input wire [DATA_WIDTH-1:0] in_93; 
    input wire [DATA_WIDTH-1:0] in_94; 
    input wire [DATA_WIDTH-1:0] in_95; 
    input wire [DATA_WIDTH-1:0] in_96; 
    input wire [DATA_WIDTH-1:0] in_97; 
    input wire [DATA_WIDTH-1:0] in_98; 
    input wire [DATA_WIDTH-1:0] in_99; 
    input wire [DATA_WIDTH-1:0] in_100; 
    input wire [6:0] sel;                  
    output reg [DATA_WIDTH-1:0] out;

    always @(*) begin
        case (sel)
            7'd1:  out = in_1;
            7'd2:  out = in_2;
            7'd3:  out = in_3;
            7'd4:  out = in_4;
            7'd5:  out = in_5;
            7'd6:  out = in_6;
            7'd7:  out = in_7;
            7'd8:  out = in_8;
            7'd9:  out = in_9;
            7'd10: out = in_10;
            7'd11: out = in_11;
            7'd12: out = in_12;
            7'd13: out = in_13;
            7'd14: out = in_14;
            7'd15: out = in_15;
            7'd16: out = in_16;
            7'd17: out = in_17;
            7'd18: out = in_18;
            7'd19: out = in_19;
            7'd20: out = in_20;
            7'd21: out = in_21;
            7'd22: out = in_22;
            7'd23: out = in_23;
            7'd24: out = in_24;
            7'd25: out = in_25;
            7'd26: out = in_26;
            7'd27: out = in_27;
            7'd28: out = in_28;
            7'd29: out = in_29;
            7'd30: out = in_30;
            7'd31: out = in_31;
            7'd32: out = in_32;
            7'd33: out = in_33;
            7'd34: out = in_34;
            7'd35: out = in_35;
            7'd36: out = in_36;
            7'd37: out = in_37;
            7'd38: out = in_38;
            7'd39: out = in_39;
            7'd40: out = in_40;
            7'd41: out = in_41;
            7'd42: out = in_42;
            7'd43: out = in_43;
            7'd44: out = in_44;
            7'd45: out = in_45;
            7'd46: out = in_46;
            7'd47: out = in_47;
            7'd48: out = in_48;
            7'd49: out = in_49;
            7'd50: out = in_50;
            7'd51: out = in_51;
            7'd52: out = in_52;
            7'd53: out = in_53;
            7'd54: out = in_54;
            7'd55: out = in_55;
            7'd56: out = in_56;
            7'd57: out = in_57;
            7'd58: out = in_58;
            7'd59: out = in_59;
            7'd60: out = in_60;
            7'd61: out = in_61;
            7'd62: out = in_62;
            7'd63: out = in_63;
            7'd64: out = in_64;
            7'd65: out = in_65;
            7'd66: out = in_66;
            7'd67: out = in_67;
            7'd68: out = in_68;
            7'd69: out = in_69;
            7'd70: out = in_70;
            7'd71: out = in_71;
            7'd72: out = in_72;
            7'd73: out = in_73;
            7'd74: out = in_74;
            7'd75: out = in_75;
            7'd76: out = in_76;
            7'd77: out = in_77;
            7'd78: out = in_78;
            7'd79: out = in_79;
            7'd80: out = in_80;
            7'd81: out = in_81;
            7'd82: out = in_82;
            7'd83: out = in_83;
            7'd84: out = in_84;
            7'd85: out = in_85;
            7'd86: out = in_86;
            7'd87: out = in_87;
            7'd88: out = in_88;
            7'd89: out = in_89;
            7'd90: out = in_90;
            7'd91: out = in_91;
            7'd92: out = in_92;
            7'd93: out = in_93;
            7'd94: out = in_94;
            7'd95: out = in_95;
            7'd96: out = in_96;
            7'd97: out = in_97;
            7'd98: out = in_98;
            7'd99: out = in_99;
            7'd100: out = in_100;
            default: out = {DATA_WIDTH{1'b0}};  // Default case
        endcase
    end

endmodule

