#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: George Booth <george.booth@kcl.ac.uk>
#
# Building on (EOMCC) work of:  Qiming Sun <osirpt.sun@gmail.com>
#                               James D. McClain
#                               Timothy Berkelbach <tim.berkelbach@gmail.com>
#
import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import rintermediates as imd
from pyscf.cc import eom_rccsd
from pyscf import __config__

def make_moms_hole(mycc, nmom_max_h, t1, t2, l1, l2, ao_repr=False, ns_def=False):
    '''
    Spin-traced one-hole moment in MO basis.

    mom_h[p,q] = \sum_{sigma} <q_sigma^\dagger (H-E_0)^n p_sigma>

    ns_def defines the terms according to the Nooijen & Snijders definitions given
    in IJQC 48 15-48 (1993). If this is not used, we get agreement with pyscf RDMs (for >2 electrons).
    Is there an error in the paper, or am I missing something in how the tensors are defined?!
    ns_def should be *false* to get correct behaviour.
        
    '''
        
    #partition : bool or str
    #    Use a matrix-partitioning for the doubles-doubles block.
    #    Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
    #    or 'full' (full diagonal elements).
    partition = getattr(__config__, 'eom_rccsd_EOM_partition', None)
    logger.info(mycc, 'partition = %s', partition)
    #print "Partition value is: ", partition
    if partition is not None: assert partition.lower() in ['mp','full']
    # ns_def breaks sum rules
    assert(not ns_def)

    nocc, nvir = t1.shape
    nmo = mycc.nmo
    dtype = np.result_type(t1, t2)
    vector_size = nocc + nocc*nocc*nvir

    imds = eom_rccsd._IMDS(mycc)
    imds.make_ip(partition)
    diag = ipccsd_diag(imds, partition)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    theta_lam = l2 * 2 - l2.transpose(1,0,2,3)

    # Construct RHS (b) vector as a_p + [a_p, T] |phi>
    # Construct LHS (e) vector as <phi|(1+L)(a_q^\dagger + [a_q^\dagger , T])
    b_vecs = np.zeros((vector_size, nmo), dtype)
    e_vecs = np.zeros((vector_size, nmo), dtype)

    # Improve this so no explicit looping
    for p in range(nocc):
        b1 = np.zeros((nocc), dtype)
        b2 = np.zeros((nocc, nocc, nvir), dtype)
        e1 = np.zeros((nocc), dtype)
        e2 = np.zeros((nocc, nocc, nvir), dtype)

        b1[p] = -1.0
        
        e1[p] = 1.
        e1 -= np.einsum('ic, c -> i', l1, t1[p,:])
        e1 -= np.einsum('jlcd, lcd -> j', l2, theta[p,:,:,:])

        if ns_def:
            # N-S result
            e2[p,:,:] = 2. * l1[:,:]
            e2[:,p,:] -= l1[:,:]
            e2 -= np.einsum('c, ijcb -> ijb', t1[p,:], theta_lam)
        else:
            # Result for consistency with pyscf cc-rdm (lambda values antisymmetrized!?)
            e2[:,p,:] = l1
            e2 -= np.einsum('c, jicb -> ijb', t1[p,:], l2)
        
        b_vecs[:,p] = amplitudes_to_vector_ip(b1, b2)
        e_vecs[:,p] = amplitudes_to_vector_ip(e1, e2)

    for p in range(nvir):
        
        b1 = -t1[:,p]
        if ns_def:
            b2 = -t2[:,:,:,p]           # This is in Nooijen paper
        else:
            b2 = -theta[:,:,:,p]
        
        e1 = l1[:,p]
        if ns_def:
            e2 = theta_lam[:,:,p,:]   # This is in Nooijen paper
        else:
            e2 = l2[:,:,:,p]   

        b_vecs[:,p+nocc] = amplitudes_to_vector_ip(b1, b2)
        e_vecs[:,p+nocc] = amplitudes_to_vector_ip(e1, e2)

    # Now, apply {\bar H} to each b
    hole_moms = [np.zeros((nmo, nmo)) for i in range(nmom_max_h+1)]
    # TODO: Would this work for complex?
    hole_moms[0] = np.dot(e_vecs.T.conj(),b_vecs)
    hole_moms[0] += hole_moms[0].conj().T
    #hole_moms[0] = -2.*np.dot(e_vecs.T.conj(),b_vecs)

    h_RHS = np.zeros_like(b_vecs)
    for h_app in range(nmom_max_h):
        if h_app == 0:
            for p in range(nmo):
                h_RHS[:,p] = ipccsd_matvec(b_vecs[:,p], imds, diag, nocc, nmo, partition)
        else:
            for p in range(nmo):
                h_RHS[:,p] = ipccsd_matvec(h_RHS[:,p], imds, diag, nocc, nmo, partition)

        hole_moms[h_app+1] = -np.dot(e_vecs.conj().T,h_RHS)
        hole_moms[h_app+1] += hole_moms[h_app+1].conj().T 

    if ao_repr:
        mo = mycc.mo_coeff
        hole_moms = [lib.einsum('pi,ij,qj->pq', mo, mom, mo.conj()) for mom in hole_moms]

    return hole_moms

def make_moms_part(mycc, nmom_max_p, t1, t2, l1, l2, ao_repr=False, ns_def=True):
    '''
    Spin-traced one-particle moment (EA) in MO basis.

    mom_h[p,q] = \sum_{sigma} <q_sigma (H-E_0)^n p_sigma^\dagger>
        
    ns_def defines the terms according to the Nooijen & Snijders definitions given
    in IJQC 48 15-48 (1993).
    ns_def should be *true* to get correct behaviour. This is different to the IP definition.
    '''
        
    #partition : bool or str
    #    Use a matrix-partitioning for the doubles-doubles block.
    #    Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
    #    or 'full' (full diagonal elements).
    partition = getattr(__config__, 'eom_rccsd_EOM_partition', None)
    logger.info(mycc, 'partition = %s', partition)
    #print "Partition value is: ", partition
    if partition is not None: assert partition.lower() in ['mp','full']
    # non-ns_def breaks sum rules
    assert(ns_def)

    nocc, nvir = t1.shape
    nmo = mycc.nmo
    dtype = np.result_type(t1, t2)
    vector_size = nvir + nocc*nvir*nvir

    imds = eom_rccsd._IMDS(mycc)
    imds.make_ea(partition)
    diag = eaccsd_diag(imds, partition)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    theta_lam = l2 * 2 - l2.transpose(1,0,2,3)

    # Construct RHS (b) vector as a_p^\dagger + [a_p^\dagger, T] |phi>
    # Construct LHS (e) vector as <phi|(1+L)(a_q + [a_q , T])
    b_vecs = np.zeros((vector_size, nmo), dtype)
    e_vecs = np.zeros((vector_size, nmo), dtype)

    # Improve this so no explicit looping
    for p in range(nocc):

        b1 = -t1[p,:].copy()
        if ns_def:
            b2 = -t2[p,:,:,:].copy()
        else:
            b2 = -theta[p,:,:,:].copy()
        e1 = l1[p,:].copy()
        if ns_def:
            e2 = 2.*l2[p,:,:,:].copy()
            e2 -= l2[:,p,:,:]
        else:
            e2 = l2[:,p,:,:]

        b_vecs[:,p] = amplitudes_to_vector_ea(b1, b2)
        e_vecs[:,p] = amplitudes_to_vector_ea(e1, e2)

    for p in range(nvir):
        b1 = np.zeros((nvir), dtype)
        b2 = np.zeros((nocc, nvir, nvir), dtype)
        e1 = np.zeros((nvir), dtype)
        e2 = np.zeros((nocc, nvir, nvir), dtype)

        b1[p] = 1.0

        e1[p] = -1.0
        e1 += np.einsum('ia,i -> a', l1,t1[:,p])
        e1 += 2.*np.einsum('klca,klc -> a',l2, t2[:,:,:,p])
        e1 -= np.einsum('klca,lkc -> a',l2, t2[:,:,:,p])

        if ns_def:
            e2[:,p,:] = -2.*l1
            e2[:,:,p] += l1
            e2 += 2.*np.einsum('k,jkba -> jab', t1[:,p], l2)
            e2 -= np.einsum('k,jkab -> jab', t1[:,p], l2)
        else:
            e2[:,:,p] = -l1
            e2 += np.einsum('k,jkab -> jab', t1[:,p], l2)

        b_vecs[:,p+nocc] = amplitudes_to_vector_ea(b1, b2)
        e_vecs[:,p+nocc] = amplitudes_to_vector_ea(e1, e2)

    # Now, apply {\bar H} to each b
    part_moms = [np.zeros((nmo, nmo)) for i in range(nmom_max_p+1)]
    # TODO: Would this work for complex?
    part_moms[0] = np.dot(e_vecs.T.conj(),b_vecs)
    part_moms[0] += part_moms[0].conj().T
    #hole_moms[0] = -2.*np.dot(e_vecs.T.conj(),b_vecs)

    p_RHS = np.zeros_like(b_vecs)
    for h_app in range(nmom_max_p):
        if h_app == 0:
            for p in range(nmo):
                p_RHS[:,p] = eaccsd_matvec(b_vecs[:,p], imds, diag, nocc, nmo, partition)
        else:
            for p in range(nmo):
                p_RHS[:,p] = eaccsd_matvec(p_RHS[:,p], imds, diag, nocc, nmo, partition)

        part_moms[h_app+1] = -np.dot(e_vecs.conj().T,p_RHS)
        part_moms[h_app+1] += part_moms[h_app+1].conj().T 

    if ao_repr:
        mo = mycc.mo_coeff
        part_moms = [lib.einsum('pi,ij,qj->pq', mo, mom, mo.conj()) for mom in part_moms]

    return part_moms

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector


def ipccsd_matvec(vector, imds, diag, nocc, nmo, partition):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kbij,k->ijb', imds.Wovoo, r1)
    # 2h1p-2h1p block
    if partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('bd,ijd->ijb', fvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', foo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', foo, r2)
    elif partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('bd,ijd->ijb', imds.Lvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', imds.Loo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', imds.Loo, r2)
        Hr2 +=  lib.einsum('klij,klb->ijb', imds.Woooo, r2)
        Hr2 += 2*lib.einsum('lbdj,ild->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('kbdj,kid->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('lbjd,ild->ijb', imds.Wovov, r2) #typo in Ref
        Hr2 +=  -lib.einsum('kbid,kjd->ijb', imds.Wovov, r2)
        tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2)
        tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2)
        Hr2 += -np.einsum('c,ijcb->ijb', tmp, imds.t2)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(imds, partition):
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype)
    for i in range(nocc):
        for j in range(nocc):
            for b in range(nvir):
                if partition == 'mp':
                    Hr2[i,j,b] += fvv[b,b]
                    Hr2[i,j,b] += -foo[i,i]
                    Hr2[i,j,b] += -foo[j,j]
                else:
                    Hr2[i,j,b] += imds.Lvv[b,b]
                    Hr2[i,j,b] += -imds.Loo[i,i]
                    Hr2[i,j,b] += -imds.Loo[j,j]
                    Hr2[i,j,b] +=  imds.Woooo[i,j,i,j]
                    Hr2[i,j,b] +=2*imds.Wovvo[j,b,b,j]
                    Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                    Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                    Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                    Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:], t2[i,j,:,b])
                    Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:], t2[i,j,:,b])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def eaccsd_matvec(vector, imds, diag, nocc, nmo, partition):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    nvir = nmo - nocc
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (30)
    # 1p-1p block
    Hr1 =  np.einsum('ac,c->a', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('ld,lad->a', 2.*imds.Fov, r2)
    Hr1 += np.einsum('ld,lda->a',   -imds.Fov, r2)
    Hr1 += np.einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    # 2p1h-2p1h block
    if partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 +=  lib.einsum('ac,jcb->jab', fvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', fvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', foo, r2)
    elif partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ea(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 +=  lib.einsum('ac,jcb->jab', imds.Lvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', imds.Lvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', imds.Loo, r2)
        Hr2 += lib.einsum('lbdj,lad->jab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
        Hr2 += -lib.einsum('lajc,lcb->jab', imds.Wovov, r2)
        Hr2 += -lib.einsum('lbcj,lca->jab', imds.Wovvo, r2)
        for a in range(nvir):
            Hr2[:,a,:] += lib.einsum('bcd,jcd->jb', imds.Wvvvv[a], r2)
        tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
        Hr2 += -np.einsum('k,kjab->jab', tmp, imds.t2)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def eaccsd_diag(imds, partition):
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape

    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = np.diag(imds.Lvv)
    Hr2 = np.zeros((nocc,nvir,nvir), dtype)
    for a in range(nvir):
        if partition != 'mp':
            _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(nvir):
            for j in range(nocc):
                if partition == 'mp':
                    Hr2[j,a,b] += fvv[a,a]
                    Hr2[j,a,b] += fvv[b,b]
                    Hr2[j,a,b] += -foo[j,j]
                else:
                    Hr2[j,a,b] += imds.Lvv[a,a]
                    Hr2[j,a,b] += imds.Lvv[b,b]
                    Hr2[j,a,b] += -imds.Loo[j,j]
                    Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                    Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                    Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                    Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                    Hr2[j,a,b] += _Wvvvva[b,a,b]
                    Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b])
                    Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b])

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

if __name__ == '__main__':
    from pyscf import gto, scf, fci, cc
    
    def create_HL_moments(n, nelec, mom_max, wavefunc, h1, h2, hl_chempot=None):
        ''' Return a list of greens function moments up to params.options['fit mom max']
            Return first the hole, then the particle list'''

        moments_h = [np.zeros((n,n)) for i in range(mom_max+1)]
        moments_p = [np.zeros((n,n)) for i in range(mom_max+1)]

        if hl_chempot is None:
             hl_chempot = 0.0

        # First for (N-1) where we have annihilated an alpha
        ci_vecs = []
        for imp in range(n):
            ci_vecs.append(fci.addons.des_a(wavefunc, n, nelec, imp))

        # Construct hamiltonian for different spin/number sector
        nelec_ = (nelec[0]-1, nelec[1])
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[1])
        h2e = fci.direct_spin1.absorb_h1e(h1, h2, n, nelec_, .5)
        
        # Now, apply the hamiltonian to each one
        for imp in range(n):
            ci = ci_vecs[imp]
            # Can get the 0th moment already
            for imp_2 in range(n):
                moments_h[0][imp_2,imp] = np.dot(ci_vecs[imp_2].flatten(),ci.flatten())
            for mom in range(mom_max):

                # Apply H-mu mom times to ci
                hci = fci.direct_spin1.contract_2e(h2e, ci, n, nelec_, (link_indexa, link_indexb))
                #hci = np.zeros_like(ci)
                ci = hci - hl_chempot*ci
                
                # when mom=n, we have applied it n+1 times
                for imp_2 in range(n):
                    moments_h[mom+1][imp_2,imp] = np.dot(ci_vecs[imp_2].flatten(),ci.flatten())
        
        # (N-1) where we have annihilated a beta
        ci_vecs = []
        for imp in range(n):
            ci_vecs.append(fci.addons.des_b(wavefunc, n, nelec, imp))

        nelec_ = (nelec[0], nelec[1]-1)
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[1])
        h2e = fci.direct_spin1.absorb_h1e(h1, h2, n, nelec_, .5)

        # Now, apply the hamiltonian to each one
        for imp in range(n):
            ci = ci_vecs[imp]
            # Can get the 0th moment already
            for imp_2 in range(n):
                moments_h[0][imp_2,imp] += np.dot(ci_vecs[imp_2].flatten(),ci.flatten())
            for mom in range(mom_max):

                # Apply H-mu mom times to ci
                hci = fci.direct_spin1.contract_2e(h2e, ci, n, nelec_, (link_indexa, link_indexb))
                ci = hci - hl_chempot*ci
                
                # when mom=n, we have applied it n+1 times
                for imp_2 in range(n):
                    moments_h[mom+1][imp_2,imp] += np.dot(ci_vecs[imp_2].flatten(),ci.flatten())

        # (N+1) where we have created an alpha
        ci_vecs = []
        for imp in range(n):
            ci_vecs.append(fci.addons.cre_a(wavefunc, n, nelec, imp))

        nelec_ = (nelec[0]+1, nelec[1])
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[1])
        h2e = fci.direct_spin1.absorb_h1e(h1, h2, n, nelec_, .5)

        # Now, apply the hamiltonian to each one
        for imp in range(n):
            ci = ci_vecs[imp]
            # Can get the 0th moment already
            for imp_2 in range(n):
                moments_p[0][imp_2,imp] = np.dot(ci_vecs[imp_2].flatten(),ci.flatten())
            for mom in range(mom_max):

                # Apply H-mu mom times to ci
                hci = fci.direct_spin1.contract_2e(h2e, ci, n, nelec_, (link_indexa, link_indexb))
                ci = hci - hl_chempot*ci
                
                # when mom=n, we have applied it n+1 times
                for imp_2 in range(n):
                    moments_p[mom+1][imp_2,imp] = np.dot(ci_vecs[imp_2].flatten(),ci.flatten())

        # (N+1) where we have created an alpha
        ci_vecs = []
        for imp in range(n):
            ci_vecs.append(fci.addons.cre_b(wavefunc, n, nelec, imp))

        nelec_ = (nelec[0], nelec[1]+1)
        link_indexa = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[0])
        link_indexb = fci.cistring.gen_linkstr_index_trilidx(range(n), nelec_[1])
        h2e = fci.direct_spin1.absorb_h1e(h1, h2, n, nelec_, .5)

        # Now, apply the hamiltonian to each one
        for imp in range(n):
            ci = ci_vecs[imp]
            # Can get the 0th moment already
            for imp_2 in range(n):
                moments_p[0][imp_2,imp] += np.dot(ci_vecs[imp_2].flatten(),ci.flatten())
            for mom in range(mom_max):

                # Apply H-mu mom times to ci
                hci = fci.direct_spin1.contract_2e(h2e, ci, n, nelec_, (link_indexa, link_indexb))
                ci = hci - hl_chempot*ci
                
                # when mom=n, we have applied it n+1 times
                for imp_2 in range(n):
                    moments_p[mom+1][imp_2,imp] += np.dot(ci_vecs[imp_2].flatten(),ci.flatten())

        return moments_h, moments_p

    mol = gto.Mole()
    mol.build(
        atom = 'H 0 0 0; H 0 0 3.5',  # in Angstrom
       #atom = 'H 0 0 0; H 0 0 2.1; H 2.1 0 0; H 2.1 0 2.1',  # in Angstrom
        basis = '6-31G',
        symmetry = False,
        verbose = 2
    )

    myhf = scf.RHF(mol)
    myhf.kernel()
#rdm_hf = myhf.make_rdm1()
    n = myhf.mo_coeff.shape[1]
    nelec = (mol.nelectron//2, mol.nelectron//2)

    cisolver = fci.FCI(mol, myhf.mo_coeff)
    e, vec = cisolver.kernel()

    rdm_fci = cisolver.make_rdm1(vec,myhf.mo_coeff.shape[1],mol.nelectron)
#print 'fci rdm: '
#print rdm_fci
    mo_ints = ao2mo.kernel(mol,myhf.mo_coeff)
    core_h_mo = reduce(np.dot,(myhf.mo_coeff.T,myhf.get_hcore(),myhf.mo_coeff))

    mom_max = 6
    fci_mom_h, fci_mom_p = create_HL_moments(n,nelec,mom_max,vec,core_h_mo,mo_ints,hl_chempot=e-mol.energy_nuc())
    assert(np.allclose(fci_mom_h[0],rdm_fci))
    assert(np.allclose(fci_mom_h[0]+fci_mom_p[0],2.0*np.eye(rdm_fci.shape[0])))

    e_mom = 0.5*np.trace(np.dot(core_h_mo,rdm_fci) - fci_mom_h[1]) + mol.energy_nuc()
    #print 'FCI energy: ',e,' Mom energy: ',e_mom,mol.energy_nuc()
    assert(np.allclose(e,e_mom))

    mycc = cc.CCSD(myhf)
    mycc.conv_tol = 1.e-12
    mycc.conv_tol_normt = 1.e-10
    mycc.max_cycle = 200
    mycc.kernel()

    l1, l2 = mycc.solve_lambda()

    dm1_cc = mycc.make_rdm1(l1=l1, l2=l2)
    CC_part_rdm = np.eye(dm1_cc.shape[0])*2.-dm1_cc
#print 'cc rdm1'
#print dm1_cc

## BOTH definitions should be exact for two-electron systems for all moments (IP) or the RDM only (EA)
## False should agree with CC RDMs for systems with > 2 electrons (IP)
## True should agree with the CC particle RDMs for systems with > 2 electrons (EA)
#    ns_def = True  
#    print 'Testing 2-e with NS definition: ',ns_def
#    mom_h = mycc.moms_hole(nmom_max_h=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    mom_p = mycc.moms_part(nmom_max_p=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    assert(np.allclose(mom_h[0],-dm1_cc))
#    assert(np.allclose(mom_p[0],-CC_part_rdm))
#    for i, hole_mom_cc in enumerate(mom_h):
#        if i == 0:
#            hole_mom_cc *= -1. 
#        assert(np.allclose(hole_mom_cc,fci_mom_h[i]))
#    assert(np.allclose(mom_p[0],-fci_mom_p[0]))
## EA-EOM-CCSD not exact for 2-electron systems apart from zeroth moment
##    for i, part_mom_cc in enumerate(mom_p):
##        if i==0:
##            part_mom_cc *= -1. 
##        assert(np.allclose(part_mom_cc,fci_mom_p[i]))
#    
#    ns_def = False 
#    print 'Testing 2-e with NS definition: ',ns_def
#    mom_h = mycc.moms_hole(nmom_max_h=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    mom_p = mycc.moms_part(nmom_max_p=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    assert(np.allclose(mom_h[0],-dm1_cc))
#    assert(np.allclose(mom_p[0],-CC_part_rdm))
#    for i, hole_mom_cc in enumerate(mom_h):
#        if i == 0:
#            hole_mom_cc *= -1. 
#        assert(np.allclose(hole_mom_cc,fci_mom_h[i]))
#    assert(np.allclose(mom_p[0],-fci_mom_p[0]))

    print 'Testing 2-e system... '
    mom_h = mycc.moms_hole(nmom_max_h=mom_max, l1=l1, l2=l2)
    mom_p = mycc.moms_part(nmom_max_p=mom_max, l1=l1, l2=l2)
    assert(np.allclose(mom_h[0],-dm1_cc))
    assert(np.allclose(mom_p[0],-CC_part_rdm))
    for i, hole_mom_cc in enumerate(mom_h):
        if i == 0:
            hole_mom_cc *= -1. 
        assert(np.allclose(hole_mom_cc,fci_mom_h[i]))
    assert(np.allclose(mom_p[0],-fci_mom_p[0]))
# EA-EOM-CCSD not exact for 2-electron systems apart from zeroth moment
#    for i, part_mom_cc in enumerate(mom_p):
#        if i==0:
#            part_mom_cc *= -1. 
#        assert(np.allclose(part_mom_cc,fci_mom_p[i]))
    print('All tests for 2e systems passed')
    
    print('Testing zeroth moments compared to CC RDMs for many-electron systems...')
    mol = gto.Mole()
    mol.build(
        atom = 'H 0 0 0; H 0 0 2.1; H 2.1 0 0; H 2.1 0 2.1',  # in Angstrom
        basis = '6-31G',
        symmetry = False,
        verbose = 2
    )

    myhf = scf.RHF(mol)
    myhf.kernel()

    mom_max = 0
    mycc = cc.CCSD(myhf)
    mycc.conv_tol = 1.e-12
    mycc.conv_tol_normt = 1.e-10
    mycc.max_cycle = 200
    mycc.kernel()

    l1, l2 = mycc.solve_lambda()

    dm1_cc = mycc.make_rdm1(l1=l1, l2=l2)
    CC_part_rdm = np.eye(dm1_cc.shape[0])*2.-dm1_cc
    print 'CCSD RDM trace: ',np.trace(dm1_cc)
    print 'CCSD part RDM trace: ',np.trace(CC_part_rdm)
#print 'cc rdm1'
#print dm1_cc

## BOTH definitions should be exact for two-electron systems for all moments (IP) or the RDM only (EA)
## False should agree with CC RDMs for systems with > 2 electrons (IP)
## True should agree with the CC particle RDMs for systems with > 2 electrons (EA)
#    ns_def = True  
#    print 'Testing many-e with NS definition: ',ns_def
#    mom_h = mycc.moms_hole(nmom_max_h=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    mom_p = mycc.moms_part(nmom_max_p=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    print 'Zeroth moment hole trace: ',np.trace(mom_h[0])
#    print 'Zeroth moment part trace: ',np.trace(mom_p[0])
#    if not np.allclose(mom_h[0],-dm1_cc):
#        print 'ERROR: Hole zeroth moment not the same as CC RDM...'
#    else:
#        print 'Hole zeroth moment same as CC RDM...'
#    if not np.allclose(mom_p[0],-CC_part_rdm):
#        print 'ERROR: Particle zeroth moment not the same as 2-CC RDM...'
#    else:
#        print 'Particle zeroth moment same as 2-CC RDM...'
#    #assert(np.allclose(mom_h[0],-dm1_cc))
#    #assert(np.allclose(mom_p[0],-CC_part_rdm))
#
#    ns_def = False 
#    print 'Testing many-e with NS definition: ',ns_def
#    mom_h = mycc.moms_hole(nmom_max_h=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    mom_p = mycc.moms_part(nmom_max_p=mom_max, l1=l1, l2=l2, ns_def=ns_def)
#    print 'Zeroth moment hole trace: ',np.trace(mom_h[0])
#    print 'Zeroth moment part trace: ',np.trace(mom_p[0])
#    if not np.allclose(mom_h[0],-dm1_cc):
#        print 'ERROR: Hole zeroth moment not the same as CC RDM...'
#    else:
#        print 'Hole zeroth moment same as CC RDM...'
#    if not np.allclose(mom_p[0],-CC_part_rdm):
#        print 'ERROR: Particle zeroth moment not the same as 2-CC RDM...'
#    else:
#        print 'Particle zeroth moment same as 2-CC RDM...'
#    #assert(np.allclose(mom_h[0],-dm1_cc))
#    #assert(np.allclose(mom_p[0],-CC_part_rdm))

    # Check sum rule for defaul moments
    print 'Testing many-e systems reproduce density matrix...'
    mom_h = mycc.moms_hole(nmom_max_h=0, l1=l1, l2=l2)
    mom_p = mycc.moms_part(nmom_max_p=0, l1=l1, l2=l2)
    assert(np.allclose(mom_h[0],-dm1_cc))
    assert(np.allclose(mom_p[0],-CC_part_rdm))

    assert(np.allclose(mom_h[0]+mom_p[0],-2.0*np.eye(mom_h[0].shape[0])))
    print 'Sum rule obeyed for default options.'
    print('All tests for many-e systems finished')
