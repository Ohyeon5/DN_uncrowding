# use pytorch from scitas step-by-step tutorial 

## in new izar cluster
> ssh choung@izer.epfl.ch
> Sinteract -p gpu -q gpu -t 01:00:00 -m 4G
> module load gcc python cuda  # gcc/8.4.0-cuda python/3.7.7 cuda/10.2.89

## OLD: in deneb2
> ssh choung@deneb2.epfl.ch
> Sinteract -p gpu -q gpu -t 01:00:00 -m 4G
> module spider py-pytorch/0.3.1
> module load (everything it's showing ex. gcc/6.4.0 openblas/0.2.20-openmp python cuda cudnn)
> module load gcc/6.4.0 openblas/0.2.20-openmp python/2.7.14 magma/2.3.0-cuda py-pytorch/0.3.1-cuda
> # please note this module is currently installed only in python2 (SCITAS team is planning to update them in Sep 2020)


