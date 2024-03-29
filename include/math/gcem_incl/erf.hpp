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
 * compile-time error function
 */

#ifndef _gcem_erf_HPP
#define _gcem_erf_HPP

// see
// http://functions.wolfram.com/GammaBetaErf/Erf/10/01/0007/

template<typename T>
constexpr
T
erf_int_cf_large_recur(const T x, const int depth)
{
    return( depth < GCEM_ERF_MAX_ITER ? \
            // if
                x + 2*depth/erf_int_cf_large_recur(x,depth+1) :
            // else
                x );
}

template<typename T>
constexpr
T
erf_int_cf_large_main(const T x)
{
    return( T(1) - T(2) * ( exp(-x*x) / T(GCEM_SQRT_PI) ) \
                / erf_int_cf_large_recur(T(2)*x,1) );
}

// see
// http://functions.wolfram.com/GammaBetaErf/Erf/10/01/0005/

template<typename T>
constexpr
T
erf_int_cf_small_recur(const T xx, const int depth)
{
    return( depth < GCEM_ERF_MAX_ITER ? \
            // if
                (2*depth - 1) - 2*xx \
                    + 4*depth*xx / erf_int_cf_small_recur(xx,depth+1) :
            // else
                (2*depth - 1) - 2*xx );
}

template<typename T>
constexpr
T
erf_int_cf_small_main(const T x)
{
    return( T(2) * x * ( exp(-x*x) / T(GCEM_SQRT_PI) ) \
                / erf_int_cf_small_recur(x*x,1) );
}

//

template<typename T>
constexpr
T
erf_int(const T x)
{
    return( x > T(2.1) ? \
            // if
                erf_int_cf_large_main(x) :
            // else
                erf_int_cf_small_main(x) );
}

template<typename T>
constexpr
T
erf_check(const T x)
{
    return( // indistinguishable from zero
            GCLIM<T>::epsilon() > abs(x) ? \
                T(0) :
            // else
                x < T(0) ? \
                    - erf_int(-x) : 
                      erf_int( x) );
}

template<typename T>
constexpr
return_t<T>
erf(const T x)
{
    return erf_check<return_t<T>>(x);
}

#endif
