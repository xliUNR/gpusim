/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
  ##
  ##   This file is part of the StatsLib C++ library.
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
 * quantile function of the Poisson distribution
 */

//
// single input

template<typename Ta, typename Tb>
statslib_constexpr
Tb
qpois_int(const Ta p, const Ta rate_par, const Ta value, const Tb count)
{
    return ( value <= p ? qpois_int(p,rate_par, ppois(count,rate_par,false), count + 1) : 
                          count - 1 );
}

template<typename Ta, typename Tb>
statslib_constexpr
Tb
qpois_check(const Ta p, const Ta rate_par)
{
    return ( STLIM<Ta>::epsilon() > p ? Tb(0) :
             //
             qpois_int<Ta,Tb>(p,rate_par,Ta(0),0U) );
}

template<typename Ta, typename Tb, typename Tc>
statslib_constexpr
Tc
qpois(const Ta p, const Tb rate_par)
{
    return qpois_check<Ta,Tc>(p,rate_par);
}

//
// matrix/vector input

template<typename Ta, typename Tb, typename Tc>
statslib_inline
void
qpois_int(const Ta* __stats_pointer_settings__ vals_in, const Tb rate_par,
                Tc* __stats_pointer_settings__ vals_out, const uint_t num_elem)
{
#ifdef STATS_USE_OPENMP
    #pragma omp parallel for
#endif
    for (uint_t j=0U; j < num_elem; j++)
    {
        vals_out[j] = qpois(vals_in[j],rate_par);
    }
}

#ifdef STATS_USE_ARMA
template<typename Ta, typename Tb, typename Tc>
statslib_inline
ArmaMat<Tc>
qpois(const ArmaMat<Ta>& X, const Tb rate_par)
{
    ArmaMat<Tc> mat_out(X.n_rows,X.n_cols);

    qpois_int<Ta,Tb,Tc>(X.memptr(),rate_par,mat_out.memptr(),mat_out.n_elem);

    return mat_out;
}
#endif

#ifdef STATS_USE_BLAZE
template<typename Ta, typename Tb, typename Tc, bool To>
statslib_inline
BlazeMat<Tc,To>
qpois(const BlazeMat<Ta,To>& X, const Tb rate_par)
{
    BlazeMat<Tc,To> mat_out(X.rows(),X.columns());

    qpois_int<Ta,Tb,Tc>(X.data(),rate_par,mat_out.data(),X.rows()*X.spacing());

    return mat_out;
}
#endif

#ifdef STATS_USE_EIGEN
template<typename Ta, typename Tb, typename Tc, int iTr, int iTc>
statslib_inline
EigMat<Tc,iTr,iTc>
qpois(const EigMat<Ta,iTr,iTc>& X, const Tb rate_par)
{
    EigMat<Tc,iTr,iTc> mat_out(X.rows(),X.cols());

    qpois_int<Ta,Tb,Tc>(X.data(),rate_par,mat_out.data(),mat_out.size());

    return mat_out;
}
#endif
