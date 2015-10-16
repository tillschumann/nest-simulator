#include "NESTNodeNeuron.h"

NESTNodeNeuron::NESTNodeNeuron()
{}
NESTNodeNeuron::NESTNodeNeuron(const float& C_m,
		   const float& Delta_T,
		   const float& E_L,
		   const float& E_ex,
		   const float& E_in,
		   const float& V_peak,
		   const float& V_reset,
		   const float& V_th,
		   const float& a,
		   const float& b,
		   const float& x,
		   const float& y,
		   const float& z,
		   const int& subnet
			      ):
		   C_m_(C_m),
		   Delta_T_(Delta_T),
		   E_L_(E_L),
		   E_ex_(E_ex),
		   E_in_(E_in),
		   V_peak_(V_peak),
		   V_reset_(V_reset),
		   V_th_(V_th),
		   a_(a),
		   b_(b),
		   x_(x),
		   y_(y),
		   z_(z),
		   subnet_(subnet)
{}

NESTNodeNeuron::~NESTNodeNeuron()
{}
void NESTNodeNeuron::set(const float& C_m,
		   const float& Delta_T,
		   const float& E_L,
		   const float& E_ex,
		   const float& E_in,
		   const float& V_peak,
		   const float& V_reset,
		   const float& V_th,
		   const float& a,
		   const float& b,
		   const float& x,
		   const float& y,
		   const float& z,
		   const int& subnet
			)
{
  C_m_= C_m;
  Delta_T_= Delta_T;
  E_L_= E_L;
  E_ex_= E_ex;
  E_in_= E_in;
  V_peak_= V_peak;
  V_reset_= V_reset;
  V_th_= V_th;
  a_= a;
  b_= b;
  x_= x;
  y_ = y;
  z_ = z;
  subnet_ = subnet;
}
void NESTNodeNeuron::serialize(double* buf)
{
  buf[0] = C_m_;
  buf[1]= Delta_T_;
  buf[2]= E_L_;
  buf[3]= E_ex_;
  buf[4]= E_in_;
  buf[5]= V_peak_;
  buf[6]= V_reset_;
  buf[7]= V_th_;
  buf[8]= a_;
  buf[9]= b_;
  buf[10] = x_;
  buf[11] = y_;
  buf[12] = z_;
  buf[13] = subnet_;
}
void NESTNodeNeuron::deserialize(double* buf)
{
  C_m_= 	buf[0];
  Delta_T_= 	buf[1];
  E_L_= 	buf[2];
  E_ex_= 	buf[3];
  E_in_= 	buf[4];
  V_peak_= 	buf[5];
  V_reset_= 	buf[6];
  V_th_= 	buf[7];
  a_= 		buf[8];
  b_= 		buf[9];
  x_=		buf[10];
  y_=		buf[11];
  z_=		buf[12];
  subnet_ = 	buf[13];
  
  //memcpy(&source_neuron_coords.x_, buf+3, sizeof(double));
  //memcpy(&source_neuron_coords.y_, buf+5, sizeof(double));
  //memcpy(&source_neuron_coords.z_, buf+7, sizeof(double));
}