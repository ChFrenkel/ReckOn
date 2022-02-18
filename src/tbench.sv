// Copyright (C) 2020-2022 University of Zurich
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// Licensed under the Solderpad Hardware License v 2.1 (the “License”); you may not use this file except in compliance
// with the License, or, at your option, the Apache License version 2.0. You may obtain a copy of the License at
// https://solderpad.org/licenses/SHL-2.1/
//
// Unless required by applicable law or agreed to in writing, any work distributed under the License is distributed on
// an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
//------------------------------------------------------------------------------
//
// "tbench.sv" -  SystemVerilog testbench including delayed-supervision navigation task implementation
// 
// Project: ReckOn - Spiking RECurrent neural network processor enabling ON-chip learning over second-long timescales
//
// Author:  C. Frenkel, Institute of Neuroinformatics, University of Zurich and ETH Zurich
//
// Cite/paper: [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network
//             processor enabling on-chip learning over second-long timescales," IEEE International Solid-State
//             Circuits Conference (ISSCC), 2022]
//
// Comments: -
//
//------------------------------------------------------------------------------

`define N             256 
`define M             8

`include "tbench.h"

`define PATH_TO_INIT_INP_WEIGHTS `"../src/data/init_rand_weights_inp_cueAcc.dat`"
`define PATH_TO_INIT_REC_WEIGHTS `"../src/data/init_rand_weights_rec_cueAcc.dat`"
`define PATH_TO_INIT_OUT_WEIGHTS `"../src/data/init_rand_weights_out_cueAcc.dat`"
`define PATH_TO_TRAIN_SET        `"../src/data/dataset_cueAcc_train.dat`"
`define PATH_TO_TEST_SET         `"../src/data/dataset_cueAcc_test.dat`"

                
module tbench ();

    /***************************
      SIGNAL DEFINITIONS
    ***************************/

    // Global inputs   -------------------------------
    logic          CLK;
    logic          CLK_INT_EN;
    logic          RST;
    
    // SPI slave       -------------------------------
    logic          SCK;
    logic          MOSI;
    logic          MISO;

    // Input bus and control inputs ------------------
    logic [`M-1:0] AERIN_ADDR;
    logic          AERIN_REQ;
    logic          AERIN_ACK;
    logic          AERIN_TAR_EN;
    logic          TARGET_VALID;
    logic          INFER_ACC;
    logic          SAMPLE;
    logic          TIME_TICK;

    // Output bus and control outputs ----------------
    logic          SPI_RDY;
    logic          TIMING_ERROR_RDY;
    logic          OUT_REQ;
    logic          OUT_ACK;
    logic [  7:0]  OUT_DATA;
    
    // Other testbench signals -----------------------
    logic                 rst_done, SPI_param_checked, SRNN_initialized_rdy, do_clk_ext;
    logic        [ 127:0] prog_val128;
    logic        [  31:0] prog_val32;
    integer               e, i, j, k;

    int fd;
    integer test;
    int status, weight;
    int correct;
    int len;
    int curr_sample;
    int curr_tick;
    int evt_neur, evt_time, evt_target;
    int inference;
    
    integer N, M;
    string path_to_init_inp_weights = `PATH_TO_INIT_INP_WEIGHTS;
    string path_to_init_rec_weights = `PATH_TO_INIT_REC_WEIGHTS;
    string path_to_init_out_weights = `PATH_TO_INIT_OUT_WEIGHTS;
    string path_to_train_set        = `PATH_TO_TRAIN_SET;
    string path_to_test_set         = `PATH_TO_TEST_SET;

    
    /***************************
      CLK
    ***************************/ 
  
    initial begin
        CLK = 1'b0; 
        wait_ns(2);
        forever begin
            wait_ns(`CLK_HALF_PERIOD);
            if (do_clk_ext)
                CLK = ~CLK; 
            else if (CLK)
                CLK = 1'b0;
        end
    end 
    
    /***************************
      RST
    ***************************/
    
    initial begin
        N = `N;
        M = `M;
        rst_done = 1'b0;
        SPI_param_checked = 1'b0;
        do_clk_ext = 1'b1;
        RST = 1'b0;
        wait_ns(50);
        RST = 1'b1;
        wait_ns(50);
        RST = 1'b0;
        wait_ns(50);
        rst_done = 1'b1;
    end
    
     
    /***************************
      Other initializations
    ***************************/
    
    initial begin
        CLK_INT_EN   = 1'b0;
        SCK          = 1'b0;
        MOSI         = 1'b0;
        AERIN_ADDR   =  'd0;
        AERIN_REQ    = 1'b0;
        AERIN_TAR_EN = 1'd0;
        TARGET_VALID = 1'b0;
        INFER_ACC    = 1'b0;
        SAMPLE       = 1'b0;
        TIME_TICK    = 1'b0;
        OUT_ACK      = 1'b0;
    end       

    
    /***************************
      STIMULI
    ***************************/ 
    
    initial begin

        // Waiting for reset
        while(!rst_done) wait_ns(10);
        
        /*****************************************************************************************************************************************************************************************************************
                                                                              PROGRAMMING THE CONTROL REGISTERS AND NEURON PARAMETERS THROUGH 32-bit SPI
        *****************************************************************************************************************************************************************************************************************/
        
        $display("----- Programming SNN parameters...");
        fork begin
            spi_send (.addr({1'b0,3'b000,12'd1,16'd0  }), .data(1                              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_EN_CONF
            spi_send (.addr({1'b0,3'b000,12'd1,16'd1  }), .data(`SPI_RO_STAGE_SEL              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_RO_STAGE_SEL
            spi_send (.addr({1'b0,3'b000,12'd1,16'd2  }), .data(`SPI_GET_CLKINT_OUT            ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_GET_CLKINT_OUT
            
            spi_send (.addr({1'b0,3'b000,12'd1,16'd8  }), .data(`SPI_RST_MODE                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_RST_MODE
            spi_send (.addr({1'b0,3'b000,12'd1,16'd9  }), .data(`SPI_DO_EPROP                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_DO_EPROP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd10 }), .data(`SPI_LOCAL_TICK                ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LOCAL_TICK
            spi_send (.addr({1'b0,3'b000,12'd1,16'd11 }), .data(`SPI_ERROR_HALT                ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_ERROR_HALT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd12 }), .data(`SPI_FP_LOC_WINP               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_WINP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd13 }), .data(`SPI_FP_LOC_WREC               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_WREC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd14 }), .data(`SPI_FP_LOC_WOUT               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_WOUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd15 }), .data(`SPI_FP_LOC_TINP               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_TINP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd16 }), .data(`SPI_FP_LOC_TREC               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_TREC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd17 }), .data(`SPI_FP_LOC_TOUT               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FP_LOC_TOUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd18 }), .data(`SPI_LEARN_SIG_SCALE           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LEARN_SIG_SCALE
            spi_send (.addr({1'b0,3'b000,12'd1,16'd19 }), .data(`SPI_REGUL_MODE                ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_MODE
            spi_send (.addr({1'b0,3'b000,12'd1,16'd20 }), .data(`SPI_REGUL_W                   ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_W
            spi_send (.addr({1'b0,3'b000,12'd1,16'd21 }), .data(`SPI_EN_STOCH_ROUND            ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_EN_STOCH_ROUND
            spi_send (.addr({1'b0,3'b000,12'd1,16'd22 }), .data(`SPI_SRAM_SPEEDMODE            ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SRAM_SPEEDMODE
            spi_send (.addr({1'b0,3'b000,12'd1,16'd23 }), .data(`SPI_TIMING_MODE               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_TIMING_MODE
            spi_send (.addr({1'b0,3'b000,12'd1,16'd25 }), .data(`SPI_REGRESSION                ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGRESSION
            spi_send (.addr({1'b0,3'b000,12'd1,16'd26 }), .data(`SPI_SINGLE_LABEL              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SINGLE_LABEL
            spi_send (.addr({1'b0,3'b000,12'd1,16'd27 }), .data(`SPI_NO_OUT_ACT                ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NO_OUT_ACT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd30 }), .data(`SPI_SEND_PER_TIMESTEP         ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEND_PER_TIMESTEP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd31 }), .data(`SPI_SEND_LABEL_ONLY           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEND_LABEL_ONLY
            spi_send (.addr({1'b0,3'b000,12'd1,16'd32 }), .data(`SPI_NOISE_EN                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NOISE_EN
            spi_send (.addr({1'b0,3'b000,12'd1,16'd33 }), .data(`SPI_FORCE_TRACES              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_FORCE_TRACES
            
            spi_send (.addr({1'b0,3'b000,12'd1,16'd64 }), .data(`SPI_CYCLES_PER_TICK           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_CYCLES_PER_TICK
            for (i=0; i<4; i=i+1) begin
                spi_send (.addr({1'b0,3'b000,12'd1,16'd65+i[1:0]}), .data((`SPI_ALPHA_CONF>>(32*i))), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_ALPHA
            end
            spi_send (.addr({1'b0,3'b000,12'd1,16'd69 }), .data(`SPI_KAPPA                     ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_KAPPA
            spi_send (.addr({1'b0,3'b000,12'd1,16'd70 }), .data(`SPI_THR_H_0                   ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_THR_H_0
            spi_send (.addr({1'b0,3'b000,12'd1,16'd71 }), .data(`SPI_THR_H_1                   ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_THR_H_1
            spi_send (.addr({1'b0,3'b000,12'd1,16'd72 }), .data(`SPI_THR_H_2                   ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_THR_H_2
            spi_send (.addr({1'b0,3'b000,12'd1,16'd73 }), .data(`SPI_THR_H_3                   ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_THR_H_3
            spi_send (.addr({1'b0,3'b000,12'd1,16'd74 }), .data(`SPI_H_0                       ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_H_0
            spi_send (.addr({1'b0,3'b000,12'd1,16'd75 }), .data(`SPI_H_1                       ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_H_1
            spi_send (.addr({1'b0,3'b000,12'd1,16'd76 }), .data(`SPI_H_2                       ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_H_2
            spi_send (.addr({1'b0,3'b000,12'd1,16'd77 }), .data(`SPI_H_3                       ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_H_3
            spi_send (.addr({1'b0,3'b000,12'd1,16'd78 }), .data(`SPI_H_4                       ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_H_4
            spi_send (.addr({1'b0,3'b000,12'd1,16'd79 }), .data(`SPI_LR_R_WINP                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_R_WINP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd80 }), .data(`SPI_LR_P_WINP                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_P_WINP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd81 }), .data(`SPI_LR_R_WREC                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_R_WREC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd82 }), .data(`SPI_LR_P_WREC                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_P_WREC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd83 }), .data(`SPI_LR_R_WOUT                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_R_WOUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd84 }), .data(`SPI_LR_P_WOUT                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_LR_P_WOUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd85 }), .data(`SPI_SEED_INP                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_INP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd86 }), .data(`SPI_SEED_REC                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_REC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd87 }), .data(`SPI_SEED_OUT                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_OUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd88 }), .data(`SPI_SEED_STRND_NEUR           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_STRND_NEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd89 }), .data(`SPI_SEED_STRND_ONEUR          ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_STRND_ONEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd90 }), .data(`SPI_SEED_STRND_TINP           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_STRND_TINP
            spi_send (.addr({1'b0,3'b000,12'd1,16'd91 }), .data(`SPI_SEED_STRND_TREC           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_STRND_TREC
            spi_send (.addr({1'b0,3'b000,12'd1,16'd92 }), .data(`SPI_SEED_STRND_TOUT           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_STRND_TOUT
            spi_send (.addr({1'b0,3'b000,12'd1,16'd93 }), .data(`SPI_SEED_NOISE_NEUR           ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_SEED_NOISE_NEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd94 }), .data(`SPI_NUM_INP_NEUR              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NUM_INP_NEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd95 }), .data(`SPI_NUM_REC_NEUR              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NUM_REC_NEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd96 }), .data(`SPI_NUM_OUT_NEUR              ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NUM_OUT_NEUR
            spi_send (.addr({1'b0,3'b000,12'd1,16'd98 }), .data(`SPI_REGUL_F0                  ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_F0
            spi_send (.addr({1'b0,3'b000,12'd1,16'd99 }), .data(`SPI_REGUL_K_INP_R             ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_K_INP_R
            spi_send (.addr({1'b0,3'b000,12'd1,16'd100}), .data(`SPI_REGUL_K_INP_P             ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_K_INP_P
            spi_send (.addr({1'b0,3'b000,12'd1,16'd101}), .data(`SPI_REGUL_K_REC_R             ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_K_REC_R
            spi_send (.addr({1'b0,3'b000,12'd1,16'd102}), .data(`SPI_REGUL_K_REC_P             ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_K_REC_P
            spi_send (.addr({1'b0,3'b000,12'd1,16'd103}), .data(`SPI_REGUL_K_MUL               ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_REGUL_K_MUL
            spi_send (.addr({1'b0,3'b000,12'd1,16'd104}), .data(`SPI_NOISE_STR                 ), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //SPI_NOISE_STR
        end join
        
        
        /*****************************************************************************************************************************************************************************************************************
                                                                                                    VERIFYING THE NEURON PARAMETERS
        *****************************************************************************************************************************************************************************************************************/        
        
        $display("----- Starting verification of programmed SNN parameters");
        assert(srnn_0.spi_slave_0.SPI_EN_CONF           == 1'b1                        ) else $fatal(0, "SPI_EN_CONF parameter not correct.");
        assert(srnn_0.spi_slave_0.SPI_RO_STAGE_SEL      == `SPI_RO_STAGE_SEL           ) else $fatal(0, "SPI_RO_STAGE_SEL parameter not correct.");
        assert(srnn_0.spi_slave_0.SPI_GET_CLKINT_OUT    == `SPI_GET_CLKINT_OUT         ) else $fatal(0, "SPI_GET_CLKINT_OUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_RST_MODE               == `SPI_RST_MODE               ) else $fatal(0, "SPI_RST_MODE parameter not correct.");
        assert(srnn_0.srnn_0.SPI_DO_EPROP               == `SPI_DO_EPROP               ) else $fatal(0, "SPI_DO_EPROP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LOCAL_TICK             == `SPI_LOCAL_TICK             ) else $fatal(0, "SPI_LOCAL_TICK parameter not correct.");
        assert(srnn_0.srnn_0.SPI_ERROR_HALT             == `SPI_ERROR_HALT             ) else $fatal(0, "SPI_ERROR_HALT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_WINP            == `SPI_FP_LOC_WINP            ) else $fatal(0, "SPI_FP_LOC_WINP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_WREC            == `SPI_FP_LOC_WREC            ) else $fatal(0, "SPI_FP_LOC_WREC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_WOUT            == `SPI_FP_LOC_WOUT            ) else $fatal(0, "SPI_FP_LOC_WOUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_TINP            == `SPI_FP_LOC_TINP            ) else $fatal(0, "SPI_FP_LOC_TINP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_TREC            == `SPI_FP_LOC_TREC            ) else $fatal(0, "SPI_FP_LOC_TREC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FP_LOC_TOUT            == `SPI_FP_LOC_TOUT            ) else $fatal(0, "SPI_FP_LOC_TOUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LEARN_SIG_SCALE        == `SPI_LEARN_SIG_SCALE        ) else $fatal(0, "SPI_LEARN_SIG_SCALE parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_MODE             == `SPI_REGUL_MODE             ) else $fatal(0, "SPI_REGUL_MODE parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_W                == `SPI_REGUL_W                ) else $fatal(0, "SPI_REGUL_W parameter not correct.");
        assert(srnn_0.srnn_0.SPI_EN_STOCH_ROUND         == `SPI_EN_STOCH_ROUND         ) else $fatal(0, "SPI_EN_STOCH_ROUND parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SRAM_SPEEDMODE         == `SPI_SRAM_SPEEDMODE         ) else $fatal(0, "SPI_SRAM_SPEEDMODE parameter not correct.");
        assert(srnn_0.srnn_0.SPI_TIMING_MODE            == `SPI_TIMING_MODE            ) else $fatal(0, "SPI_TIMING_MODE parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGRESSION             == `SPI_REGRESSION             ) else $fatal(0, "SPI_REGRESSION parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SINGLE_LABEL           == `SPI_SINGLE_LABEL           ) else $fatal(0, "SPI_SINGLE_LABEL parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NO_OUT_ACT             == `SPI_NO_OUT_ACT             ) else $fatal(0, "SPI_NO_OUT_ACT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEND_PER_TIMESTEP      == `SPI_SEND_PER_TIMESTEP      ) else $fatal(0, "SPI_SEND_PER_TIMESTEP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEND_LABEL_ONLY        == `SPI_SEND_LABEL_ONLY        ) else $fatal(0, "SPI_SEND_LABEL_ONLY parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NOISE_EN               == `SPI_NOISE_EN               ) else $fatal(0, "SPI_NOISE_EN parameter not correct.");
        assert(srnn_0.srnn_0.SPI_FORCE_TRACES           == `SPI_FORCE_TRACES           ) else $fatal(0, "SPI_FORCE_TRACES parameter not correct.");
        
        assert(srnn_0.srnn_0.SPI_CYCLES_PER_TICK        == `SPI_CYCLES_PER_TICK        ) else $fatal(0, "SPI_CYCLES_PER_TICK parameter not correct.");
        assert(srnn_0.srnn_0.SPI_ALPHA_CONF             == `SPI_ALPHA_CONF             ) else $fatal(0, "SPI_ALPHA_CONF parameter not correct.");
        assert(srnn_0.srnn_0.SPI_KAPPA                  == `SPI_KAPPA                  ) else $fatal(0, "SPI_KAPPA parameter not correct.");
        assert(srnn_0.srnn_0.SPI_THR_H_0                == `SPI_THR_H_0                ) else $fatal(0, "SPI_THR_H_0 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_THR_H_1                == `SPI_THR_H_1                ) else $fatal(0, "SPI_THR_H_1 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_THR_H_2                == `SPI_THR_H_2                ) else $fatal(0, "SPI_THR_H_2 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_THR_H_3                == `SPI_THR_H_3                ) else $fatal(0, "SPI_THR_H_3 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_H_0                    == `SPI_H_0                    ) else $fatal(0, "SPI_H_0 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_H_1                    == `SPI_H_1                    ) else $fatal(0, "SPI_H_1 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_H_2                    == `SPI_H_2                    ) else $fatal(0, "SPI_H_2 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_H_3                    == `SPI_H_3                    ) else $fatal(0, "SPI_H_3 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_H_4                    == `SPI_H_4                    ) else $fatal(0, "SPI_H_4 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_R_WINP              == `SPI_LR_R_WINP              ) else $fatal(0, "SPI_LR_R_WINP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_P_WINP              == `SPI_LR_P_WINP              ) else $fatal(0, "SPI_LR_P_WINP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_R_WREC              == `SPI_LR_R_WREC              ) else $fatal(0, "SPI_LR_R_WREC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_P_WREC              == `SPI_LR_P_WREC              ) else $fatal(0, "SPI_LR_P_WREC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_R_WOUT              == `SPI_LR_R_WOUT              ) else $fatal(0, "SPI_LR_R_WOUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_LR_P_WOUT              == `SPI_LR_P_WOUT              ) else $fatal(0, "SPI_LR_P_WOUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_INP               == `SPI_SEED_INP               ) else $fatal(0, "SPI_SEED_INP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_REC               == `SPI_SEED_REC               ) else $fatal(0, "SPI_SEED_REC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_OUT               == `SPI_SEED_OUT               ) else $fatal(0, "SPI_SEED_OUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_STRND_NEUR        == `SPI_SEED_STRND_NEUR        ) else $fatal(0, "SPI_SEED_STRND_NEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_STRND_ONEUR       == `SPI_SEED_STRND_ONEUR       ) else $fatal(0, "SPI_SEED_STRND_ONEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_STRND_TINP        == `SPI_SEED_STRND_TINP        ) else $fatal(0, "SPI_SEED_STRND_TINP parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_STRND_TREC        == `SPI_SEED_STRND_TREC        ) else $fatal(0, "SPI_SEED_STRND_TREC parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_STRND_TOUT        == `SPI_SEED_STRND_TOUT        ) else $fatal(0, "SPI_SEED_STRND_TOUT parameter not correct.");
        assert(srnn_0.srnn_0.SPI_SEED_NOISE_NEUR        == `SPI_SEED_NOISE_NEUR        ) else $fatal(0, "SPI_SEED_NOISE_NEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NUM_INP_NEUR           == `SPI_NUM_INP_NEUR           ) else $fatal(0, "SPI_NUM_INP_NEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NUM_REC_NEUR           == `SPI_NUM_REC_NEUR           ) else $fatal(0, "SPI_NUM_REC_NEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NUM_OUT_NEUR           == `SPI_NUM_OUT_NEUR           ) else $fatal(0, "SPI_NUM_OUT_NEUR parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_F0               == `SPI_REGUL_F0               ) else $fatal(0, "SPI_REGUL_F0 parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_K_INP_R          == `SPI_REGUL_K_INP_R          ) else $fatal(0, "SPI_REGUL_K_INP_R parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_K_INP_P          == `SPI_REGUL_K_INP_P          ) else $fatal(0, "SPI_REGUL_K_INP_P parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_K_REC_R          == `SPI_REGUL_K_REC_R          ) else $fatal(0, "SPI_REGUL_K_REC_R parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_K_REC_P          == `SPI_REGUL_K_REC_P          ) else $fatal(0, "SPI_REGUL_K_REC_P parameter not correct.");
        assert(srnn_0.srnn_0.SPI_REGUL_K_MUL            == `SPI_REGUL_K_MUL            ) else $fatal(0, "SPI_REGUL_K_MUL parameter not correct.");
        assert(srnn_0.srnn_0.SPI_NOISE_STR              == `SPI_NOISE_STR              ) else $fatal(0, "SPI_NOISE_STR parameter not correct.");

        $display("----- Done verification of programmed SNN parameters.");
        spi_send (.addr({1'b0,3'b000,12'd1,16'd0}), .data(0), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); //Disable SPI_EN_CONF
        
        wait_ns(1000);


        /*****************************************************************************************************************************************************************************************************************
                                                                                            FAST DELAYED-SUPERVISION NAVIGATION E-PROP BENCHMARKING
        *****************************************************************************************************************************************************************************************************************/

        // Enter programming phase
        spi_send (.addr({1'b0,3'h0,12'd1,16'd0}), .data(1), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Enable SPI_EN_CONF

        $display("----- Starting delayed-supervision navigation benchmarking (e-prop enabled on random weights)...");
        $display("      Initializing input weights (with SPI)...");
        fd = $fopen(path_to_init_inp_weights,"r");
        fork begin
            for (i=0; i<40; i=i+1) begin
                spi_half_w(.data({1'b0,3'h3,12'd25,{2'b0,i[7:0],6'd0}}), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Header of a multi-write transaction starting at postsynaptic address 0 of pre-synaptic neuron i (25 32-bit writes = 100 post-synaptic weights)
                for (j=0; j<25; j=j+1) begin
                    prog_val32 = 32'b0;
                    for (k=0; k<4; k=k+1) begin
                        status = $fscanf(fd,"%d",weight); 
                        assert(status == 1) else $fatal(0, "A problem occured while processing input weights file.");
                        prog_val32 = prog_val32 | ({24'b0,weight[7:0]} << (8*k));
                    end
                    spi_half_w(.data(prog_val32), .MISO(MISO), .MOSI(MOSI), .SCK(SCK));
                end
            end
        end join
        $fclose(fd);
        $display("      Initializing recurrent weights (with SPI)...");
        fd = $fopen(path_to_init_rec_weights,"r");
        fork begin
            for (i=0; i<100; i=i+1) begin
                spi_half_w(.data({1'b0,3'h4,12'd25,{2'b0,i[7:0],6'd0}}), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Header of a multi-write transaction starting at postsynaptic address 0 of pre-synaptic neuron i (25 32-bit writes = 100 post-synaptic weights)
                for (j=0; j<25; j=j+1) begin
                    prog_val32 = 32'b0;
                    for (k=0; k<4; k=k+1) begin
                        status = $fscanf(fd,"%d",weight); 
                        assert(status == 1) else $fatal(0, "A problem occured while processing recurrent weights file.");
                        prog_val32 = prog_val32 | ({24'b0,weight[7:0]} << (8*k));
                    end
                    spi_half_w(.data(prog_val32), .MISO(MISO), .MOSI(MOSI), .SCK(SCK));
                end
            end
        end join
        $fclose(fd);
        $display("      Initializing output weights (with SPI)...");
        fd = $fopen(path_to_init_out_weights,"r");
        fork begin
            for (i=0; i<100; i=i+1) begin
                prog_val32 = 32'b0;
                for (k=0; k<2; k=k+1) begin
                    status = $fscanf(fd,"%d",weight); 
                    assert(status == 1) else $fatal(0, "A problem occured while processing output weights file.");
                    prog_val32 = prog_val32 | ({24'b0,weight[7:0]} << (8*k));
                end
                spi_send (.addr({1'b0,3'h5,12'd1,{6'b0,i[7:0],2'b0}}), .data(prog_val32), .MISO(MISO), .MOSI(MOSI), .SCK(SCK));
            end
        end join
        $fclose(fd);
        wait_ns(1000);

        $display("      Initializing neurons (with SPI)...");
        fork begin
            prog_val128 = {12'hFEF,16'd614,{2{10'b0,12'b0,12'b0,16'b0}}};
            spi_half_w(.data({1'b0,3'h1,12'd400,16'd0}), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Header of a multi-write transaction starting at address 0 of neurons
            for (i=0; i<100; i=i+1) begin
                for (j=0; j<4; j=j+1) begin
                    prog_val32 = prog_val128[j*32+:32];
                    spi_half_w(.data(prog_val32), .MISO(MISO), .MOSI(MOSI), .SCK(SCK));
                end
            end
            spi_send (.addr({1'b0,3'h0,12'd1,16'd0}), .data(0), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Disable SPI_EN_CONF
        end join
        wait_ns(1000);


        $display("      Start training for %d epochs...", `EPOCHS);
        for (e=0; e<`EPOCHS; e++) begin // For a given number of epochs

            $display("      --- Epoch %d:", e+1);
            for (test=0; test<=1; test++) begin

                correct = 0;
                fork begin
                    spi_send (.addr({1'b0,3'h0,12'd1,16'd0}), .data(1), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Enable SPI_EN_CONF
                    if (test) begin             // Test
                        spi_send (.addr({1'b0,3'h0,12'd1,16'd9}), .data(3'b000), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // SPI_DO_EPROP all disabled
                        fd = $fopen(path_to_test_set,"r");
                    end else begin              // Train
                        spi_send (.addr({1'b0,3'h0,12'd1,16'd9}), .data(3'b111), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // SPI_DO_EPROP all enabled
                        fd = $fopen(path_to_train_set,"r");
                    end
                    spi_send (.addr({1'b0,3'h0,12'd1,16'd0}), .data(0), .MISO(MISO), .MOSI(MOSI), .SCK(SCK)); // Disable SPI_EN_CONF
                end join
                status = $fscanf(fd,"%d",len); 
                assert (status == 1) else $fatal(0, "A problem occured while reading the header of the delayed-supervision navigation dataset.");
                wait_ns(1000);

                for (curr_sample=0; curr_sample<len; curr_sample=curr_sample+1) begin
                    curr_tick = 0;
                    SAMPLE = 1'b1;
                    wait_ns(100);
                    while (1) begin
                        status = $fscanf(fd, "%d, %d", evt_neur, evt_time); 
                        assert(status == 2) else $fatal(0, "A problem occured while reading an event from the dataset.");
                        if (evt_neur == -2) begin           // Target start mark
                            evt_target = evt_time;
                            while (curr_tick < 2100) begin
                                TIME_TICK = 1'b1;
                                wait_ns(100);
                                TIME_TICK = 1'b0;
                                curr_tick = curr_tick + 1;
                                @(posedge TIMING_ERROR_RDY);
                                wait_ns(100);
                            end
                            if (!test) begin
                                AERIN_TAR_EN = 1'b1;
                                aer_send(.addr_in(evt_target[7:0]), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
                                wait_ns(200);
                                AERIN_TAR_EN = 1'b0;
                                wait_ns(100);
                                TARGET_VALID = 1'b1;
                            end
                            INFER_ACC    = 1'b1;
                            wait_ns(100);
                            status = $fscanf(fd, "%d, %d", evt_neur, evt_time); 
                            assert(status == 2) else $fatal(0, "A problem occured while reading an event post-target from the dataset.");
                        end 
                        while (curr_tick < evt_time) begin
                            if (curr_tick == 2249) begin
                                SAMPLE = 1'b0;
                                wait_ns(100);
                            end
                            TIME_TICK = 1'b1;
                            wait_ns(100);
                            TIME_TICK = 1'b0;

                            if (curr_tick == 2249)
                                break;
                            curr_tick = curr_tick + 1;
                            @(posedge TIMING_ERROR_RDY);
                            wait_ns(100);

                        end if (evt_neur == -1)
                            break;
                        aer_send(.addr_in(evt_neur[7:0]), .addr_out(AERIN_ADDR), .ack(AERIN_ACK), .req(AERIN_REQ));
                        wait_ns(100);
                    end

                    while (!OUT_REQ)
                        wait_ns(100);
                    inference = OUT_DATA[3:0];
                    if (inference == evt_target[3:0])
                        correct = correct + 1;
                    $display("            Sample %2d: inference is %2d, label is %2d", curr_sample, inference, evt_target); 
                    OUT_ACK = 1'b1;
                    while (OUT_REQ)
                        wait_ns(100);
                    OUT_ACK = 1'b0;
                    wait_ns(1000);
                    TARGET_VALID = 1'b0;
                    INFER_ACC    = 1'b0;
                    wait_ns(1000);
                end
                $fclose(fd);
                if (test)
                    $display("          Score on test set is %0d/%0d (%d percents)!", correct, len, 100*correct/len);
                else
                    $display("          Score while training is %0d/%0d (%d percents)!", correct, len, 100*correct/len);
            end
        end

        $display("----- Ending delayed-supervision navigation benchmarking.");


        /***************************
         Properly ending simulation
        ***************************/     
       
        wait_ns(10000);
        $finish;

    end 
    

    /***************************
      SNN INSTANTIATION
    ***************************/

    reckon srnn_0 (
        // Global inputs   -------------------------------
        .CLK_EXT(CLK),
        .CLK_INT_EN(CLK_INT_EN),
        .RST(RST),
        
        // SPI slave       -------------------------------
        .SCK(SCK),
        .MOSI(MOSI),
        .MISO(MISO),

        // Input bus and control inputs ------------------
        .AERIN_ADDR(AERIN_ADDR),
        .AERIN_REQ(AERIN_REQ),
        .AERIN_ACK(AERIN_ACK),
        .AERIN_TAR_EN(AERIN_TAR_EN),
        .SAMPLE(SAMPLE),
        .TIME_TICK(TIME_TICK),
        .TARGET_VALID(TARGET_VALID),
        .INFER_ACC(INFER_ACC),

        // Output bus and control outputs ----------------
        .SPI_RDY(SPI_RDY),
        .TIMING_ERROR_RDY(TIMING_ERROR_RDY),
        .OUT_REQ(OUT_REQ),
        .OUT_ACK(OUT_ACK),
        .OUT_DATA(OUT_DATA)
    );
    
    
    /***********************************************************************
                            TASK IMPLEMENTATIONS
    ************************************************************************/ 

    /***************************
     SIMPLE TIME-HANDLING TASKS
    ***************************/
    
    // Based on a correct definition of the simulator timescale at 1ns/1ps.
    task wait_ns;
        input   tics_ns;
        integer tics_ns;
        #tics_ns;
    endtask
    
    /***************************
     AER send event
    ***************************/
    
    task automatic aer_send (
        input  logic [`M-1:0] addr_in,
        ref    logic [`M-1:0] addr_out,
        ref    logic        ack,
        ref    logic        req
    );
        while (ack) wait_ns(1);
        addr_out = addr_in;
        wait_ns(1);
        req = 1'b1;
        while (!ack) wait_ns(1);
        wait_ns(1);
        req = 1'b0;
    endtask
    
    /***************************
     SPI half transaction (for multi-R/W)
    ***************************/

    task automatic spi_half_w (
        input  logic [31:0] data,
        input  logic        MISO, // not used
        ref    logic        MOSI,
        ref    logic        SCK
    );
        integer i;
        
        for (i=0; i<32; i=i+1) begin
            MOSI = data[31-i];
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
    endtask

    task automatic spi_half_r (
        output logic [31:0] data,
        ref    logic        MISO,
        ref    logic        MOSI,
        ref    logic        SCK
    );
        integer i;

        for (i=0; i<32; i=i+1) begin
            wait_ns(`SCK_HALF_PERIOD);
            data = {data[30:0],MISO};
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
    endtask
    
    /***************************
     SPI send data
    ***************************/

    task automatic spi_send (
        input  logic [31:0] addr,
        input  logic [31:0] data,
        input  logic        MISO, // not used
        ref    logic        MOSI,
        ref    logic        SCK
    );
        integer i;
        
        for (i=0; i<32; i=i+1) begin
            MOSI = addr[31-i];
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
        for (i=0; i<32; i=i+1) begin
            MOSI = data[31-i];
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
    endtask
    
    /***************************
     SPI read data
    ***************************/

    task automatic spi_read (
        input  logic [31:0] addr,
        output logic [31:0] data,
        ref    logic        MISO,
        ref    logic        MOSI,
        ref    logic        SCK
    );
        integer i;
        
        for (i=0; i<32; i=i+1) begin
            MOSI = addr[31-i];
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
        for (i=0; i<32; i=i+1) begin
            wait_ns(`SCK_HALF_PERIOD);
            data = {data[30:0],MISO};
            SCK  = 1'b1;
            wait_ns(`SCK_HALF_PERIOD);
            SCK  = 1'b0;
        end
    endtask
    
    
endmodule


