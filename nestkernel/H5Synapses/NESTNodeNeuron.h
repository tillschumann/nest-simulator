#include <iostream>

#ifndef NESTNODENEURON_CLASS
#define NESTNODENEURON_CLASS



class NESTNodeNeuron
{
private:
public:  
    NESTNodeNeuron();
    NESTNodeNeuron(const float& C_m,
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
		  );
    ~NESTNodeNeuron();
    
    std::vector< double > prop_values_;
    
    float C_m_;
    float Delta_T_;
    float E_L_;
    float E_ex_;
    float E_in_;
    float V_peak_;
    float V_reset_;
    float V_th_;
    float a_;
    float b_;
    float x_;
    float y_;
    float z_;
    
    
    //g_L
    //tau_w
    //t_ref
    //tau_syn_ex
    //tau_syn_in
    
    int subnet_;
    
    void set( const float& C_m,
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
	    );
    
    void serialize(double* buf);
    void deserialize(double* buf);
    
    //bool operator<(const NESTNodeNeuron& rhs) const;
};



#endif