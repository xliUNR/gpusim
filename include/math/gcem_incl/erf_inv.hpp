/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/* 
 * compile-time inverse error function
 *
 * Initial approximation based on:
 * 'Approximating the erfinv function' by Mike Giles
 */

#ifndef _gcem_erf_inv_HPP
#define _gcem_erf_inv_HPP

template<typename T>
constexpr T erf_inv_decision(const T value, const T p, const T direc, const int iter_count);

//
// initial value

// two cases: (1) a < 5; and (2) otherwise

template<typename T>
constexpr
T
erf_inv_initial_val_coef_2(const T a, const T p_term, const int order)
{
    return( order == 1 ? T(-0.000200214257L) :
            order == 2 ? T( 0.000100950558L) + a*p_term :
            order == 3 ? T( 0.00134934322L)  + a*p_term :
            order == 4 ? T(-0.003673428440L) + a*p_term :
            order == 5 ? T( 0.005739507730L) + a*p_term :
            order == 6 ? T(-0.00762246130L)  + a*p_term :
            order == 7 ? T( 0.009438870470L) + a*p_term :
            order == 8 ? T( 1.001674060000L) + a*p_term :
            order == 9 ? T( 2.83297682000L)  + a*p_term :
                         p_term );
}

template<typename T>
constexpr
T
erf_inv_initial_val_case_2(const T a, const T p_term, const int order)
{
    return( order == 9 ? \
            // if
                erf_inv_initial_val_coef_2(a,p_term,order) :
            // else
                erf_inv_initial_val_case_2(a,erf_inv_initial_val_coef_2(a,p_term,order),order+1) );
}

template<typename T>
constexpr
T
erf_inv_initial_val_coef_1(const T a, const T p_term, const int order)
{
    return( order == 1 ? T( 2.81022636e-08L) : 
            order == 2 ? T( 3.43273939e-07L) + a*p_term :
            order == 3 ? T(-3.5233877e-06L)  + a*p_term :
            order == 4 ? T(-4.39150654e-06L) + a*p_term :
            order == 5 ? T( 0.00021858087L)  + a*p_term :
            order == 6 ? T(-0.00125372503L)  + a*p_term :
            order == 7 ? T(-0.004177681640L) + a*p_term :
            order == 8 ? T( 0.24664072700L)  + a*p_term :
            order == 9 ? T( 1.50140941000L)  + a*p_term :
                         p_term );
}

template<typename T>
constexpr
T
erf_inv_initial_val_case_1(const T a, const T p_term, const int order)
{
    return( order == 9 ? \
            // if
                erf_inv_initial_val_coef_1(a,p_term,order) :
            // else
                erf_inv_initial_val_case_1(a,erf_inv_initial_val_coef_1(a,p_term,order),order+1) );
}

template<typename T>
constexpr
T
erf_inv_initial_val_int(const T a)
{
    return( a < T(5) ? \
            // if
                erf_inv_initial_val_case_1(a-T(2.5),T(0),1) :
            // else
                erf_inv_initial_val_case_2(sqrt(a)-T(3),T(0),1) );
}

template<typename T>
constexpr
T
erf_inv_initial_val(const T x)
{
    return x*erf_inv_initial_val_int( -log( (T(1) - x)*(T(1) + x) ) );
}

//
// Halley recursion

template<typename T>
constexpr
T
erf_inv_err_val(const T value, const T p)
{   // err_val = f(x)
    return( erf(value) - p );
}

template<typename T>
constexpr
T
erf_inv_deriv_1(const T value)
{   // derivative of the error function w.r.t. x
    return( exp( -value*value ) );
}

template<typename T>
constexpr
T
erf_inv_deriv_2(const T value, const T deriv_1)
{   // second derivative of the error function w.r.t. x
    return( deriv_1*( -T(2)*value ) );
}

template<typename T>
constexpr
T
erf_inv_ratio_val_1(const T value, const T p, const T deriv_1)
{
    return( erf_inv_err_val(value,p) / deriv_1 );
}

template<typename T>
constexpr
T
erf_inv_ratio_val_2(const T value, const T p, const T deriv_1)
{
    return( erf_inv_deriv_2(value,deriv_1) / deriv_1 );
}

template<typename T>
constexpr
T
erf_inv_halley(const T ratio_val_1, const T ratio_val_2)
{
    return( ratio_val_1 / max( T(0.8), min( T(1.2), T(1) - T(0.5)*ratio_val_1*ratio_val_2 ) ) );
}

template<typename T>
constexpr
T
erf_inv_recur(const T value, const T p, const T deriv_1, const int iter_count)
{
    return erf_inv_decision( value, p, 
                             erf_inv_halley(erf_inv_ratio_val_1(value,p,deriv_1), 
                                            erf_inv_ratio_val_2(value,p,deriv_1)),
                             iter_count );
}

template<typename T>
constexpr
T
erf_inv_decision(const T value, const T p, const T direc, const int iter_count)
{
    return( iter_count < GCEM_ERF_INV_MAX_ITER ? \
            // if
                erf_inv_recur(value-direc,p, erf_inv_deriv_1(value), iter_count+1) :
            // else
                value - direc );
}

template<typename T>
constexpr
T
erf_inv_start(const T initial_val, const T p)
{
    return erf_inv_recur(initial_val,p,erf_inv_deriv_1(initial_val),1);
}

template<typename T>
constexpr
T
erf_inv_int(const T p)
{
    return erf_inv_start(erf_inv_initial_val(p),p);
}

template<typename T>
constexpr
return_t<T>
erf_inv(const T p)
{
    return erf_inv_int<return_t<T>>(p);
}


#endif
