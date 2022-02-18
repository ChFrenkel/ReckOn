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
// "tbench.h" -  Testbench header with SPI parameters for fast-convergence delayed-supervision navigation task
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

`define CLK_HALF_PERIOD             10
`define SCK_HALF_PERIOD             50

`define EPOCHS						10

`define SPI_RO_STAGE_SEL            9'b0
`define SPI_GET_CLKINT_OUT          1'b0

`define SPI_RST_MODE                1'b0
`define SPI_DO_EPROP                3'b000
`define SPI_LOCAL_TICK              1'b0
`define SPI_ERROR_HALT              1'b1
`define SPI_FP_LOC_WINP             3'b011
`define SPI_FP_LOC_WREC             3'b011
`define SPI_FP_LOC_WOUT             3'b011
`define SPI_FP_LOC_TINP             3'b011
`define SPI_FP_LOC_TREC             3'b100
`define SPI_FP_LOC_TOUT             3'b110
`define SPI_LEARN_SIG_SCALE         4'b0101
`define SPI_REGUL_MODE              3'b010
`define SPI_REGUL_W                 2'b11
`define SPI_EN_STOCH_ROUND          1'b1
`define SPI_SRAM_SPEEDMODE          8'b00000000
`define SPI_TIMING_MODE             1'b0
`define SPI_REGRESSION              1'b0
`define SPI_SINGLE_LABEL            1'b1
`define SPI_NO_OUT_ACT              1'b0
`define SPI_SEND_PER_TIMESTEP       1'b0
`define SPI_SEND_LABEL_ONLY         1'b1
`define SPI_NOISE_EN                1'b0
`define SPI_FORCE_TRACES            1'b0

`define SPI_CYCLES_PER_TICK         32'b0
`define SPI_ALPHA_CONF              128'h0
`define SPI_KAPPA                   8'h79
`define SPI_THR_H_0                 $signed(16'd205)
`define SPI_THR_H_1                 $signed(16'd205)
`define SPI_THR_H_2                 $signed(16'd205)
`define SPI_THR_H_3                 $signed(16'd205)
`define SPI_H_0                     $signed(5'd0)
`define SPI_H_1                     $signed(5'd0)
`define SPI_H_2                     $signed(5'd0)
`define SPI_H_3                     $signed(5'd0)
`define SPI_H_4                     $signed(5'd1)
`define SPI_LR_R_WINP               5'b0 
`define SPI_LR_P_WINP               5'd8
`define SPI_LR_R_WREC               5'b0
`define SPI_LR_P_WREC               5'd8
`define SPI_LR_R_WOUT               5'b0
`define SPI_LR_P_WOUT               5'd14
`define SPI_SEED_INP                25'hF0F0
`define SPI_SEED_REC                25'hF1F1
`define SPI_SEED_OUT                22'hF2F2
`define SPI_SEED_STRND_NEUR         30'h3F3FF3F3
`define SPI_SEED_STRND_ONEUR        15'hF4F4
`define SPI_SEED_STRND_TINP         30'h3F5FF5F5
`define SPI_SEED_STRND_TREC         30'h3F6FF6F6
`define SPI_SEED_STRND_TOUT         30'h3F7FF7F7
`define SPI_SEED_NOISE_NEUR         17'h00FF0
`define SPI_NUM_INP_NEUR            8'd39
`define SPI_NUM_REC_NEUR            8'd99
`define SPI_NUM_OUT_NEUR            4'd1
`define SPI_REGUL_F0                12'd160
`define SPI_REGUL_K_INP_R           5'b0
`define SPI_REGUL_K_INP_P           5'd10
`define SPI_REGUL_K_REC_R           5'b0
`define SPI_REGUL_K_REC_P           5'd10
`define SPI_REGUL_K_MUL             5'd0
`define SPI_NOISE_STR               4'b0000
