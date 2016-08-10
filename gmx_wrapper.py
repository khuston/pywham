"""
    Gromacs wrapper
    Kyle J. Huston
    22 February 2016

    Read class Gromacs docstring for info
"""
from subprocess import call, Popen, PIPE

def regularize_args(args):
    args = [str(arg) for arg in args]
    args = ' '.join(args)
    return args.split()

class GromacsError(Exception):
    pass

class Gromacs:
    """
        Wrapper for Gromacs
                                                                                   
          What this class is for
        - Handles program calling differences between versions 4 and 5
        - Raises exceptions of type GromacsError when Gromacs exits in error
        - TODO: test nohup flag and handling
        - TODO: print stdout and stderr to screen for verbosity==2
        - TODO: Handle interactive programs
        - There are other versions differences, especially with mdp files.
          This does not handle those differences, because all this wrapper does is
          make program calls with arguments. At most, it may deal with program
          interaction (e.g. g_energy, make_ndx), but it won't make topology or
          mdp files.
        - More complex interactions will have to be handled by other modules,
          e.g. gmx_util

        Example usage:
                                                                         
          from gmx_wrapper import Gromacs
          from gmx_util import AppendToTop
                                                                         
          gmx = Gromacs(5.1)
          gmx.solvate('-box 4 4 4 -o water.pdb')
          AppendToTop('template.top','water.pdb')
          gmx.grompp('-f md.mdp -c water.pdb -p template.top -o md.tpr')
          gmx.mdrun('-deffnm md -nt %d'%(num_threads))
    """
    _version_key    = {5.1:5,4.6:4,5:5,4:4}
    _mdrun = {5:['gmx','mdrun'], 4:['mdrun']}
    _mdrun_mpi = {5:['mpirun','-np','1','gmx_mpi','mdrun'], 4:['mpirun','-np','1','mdrun_mpi']}
    _grompp = {5:['gmx','grompp'], 4:['grompp']}
    _genbox = {5:['gmx','solvate'], 4:['genbox']}
    _editconf = {5:['gmx','editconf'], 4:['editconf']}
    _genconf = {5:['gmx','genconf'], 4:['genconf']}
    _trjcat = {5:['gmx','trjcat'], 4:['trjcat']}
    _trjconv = {5:['gmx','trjconv'], 4:['trjconv']}
    _energy = {5:['gmx','energy'], 4:['energy']}
    _gyrate = {5:['gmx','gyrate'], 4:['g_gyrate']}
    _wham = {5:['gmx','wham'], 4:['g_wham']}
    _dump = {5:['gmx','dump'], 4:['gmxdump']}
    _tpbconv = {5:['gmx','convert-tpr'], 4:['tpbconv']}

    def __init__(self, version='5.1', verbosity=0, ignore_pme_warning=False, ignore_cg_radii_warning=False, ignore_pr_warning=False):
        """
            Inialize gromacs wrapper with version (default 5.1)

            version: version of Gromacs to use
            verbosity:
                0 - No output
                1 - Print gmx command line
                2 - TODO
            ignore_pme_warning (False): if True then ignore pme warning,
            increment maxwarn and continue
            ignore_cg_radii_warning (False): if True then ignore cg radii
            warning, increment maxwarn and continue
            ignore_pr_warning (False): if True then ignore gen-vel for
            Parrinello-Rahman thermostat warning, increment maxwarn and
            continue
        """
        # Truncate version beyond second point
        self.verbosity = int(verbosity)
        self.ignore_pme_warning = ignore_pme_warning
        self.ignore_cg_radii_warning = ignore_cg_radii_warning
        self.ignore_pr_warning = ignore_pr_warning
        nums = str(version).split('.')
        version = float('%s.%s'%(nums[0],nums[1]))
        if version < 5:
            self.version = 4
        elif version == 5.1:
            self.version = version
        else:
            raise Exception('Version %f not supported!'%(version))
        self.key = self._version_key[version]
        self.test_availability()

    def __repr__(self):
        return "<Gromacs version {0:} wrapper>".format(self.version)

    def test_availability(self):
        """
            See if gromacs calls will work
        """
        try:
            self.mdrun()
        except IOError as e:
            print('Couldn\'t open mdrun')
            raise e
        except GromacsError:
            pass

    def check_warnings(self,stderr_path=None):
        """
            Read standard error file at stderr_path and/or
            interpret err_val to look for warnings.
            Prints warnings if they occur.
        """
        warnings = []
        with open(stderr_path,'r') as stderr:
            warning_flag = False
            for line in stderr.readlines():
                if 'WARNING' in line and not warning_flag:
                    warning_flag = True
                    warning_msg = line
                elif 'WARNING' in line and warning_flag:
                    print('Found WARNING without blank line preceeding')
                    print('I don\'t handle these yet')
                elif warning_flag and line.strip():
                    warning_msg += line
                elif warning_flag and not line.strip():
                    warning_flag = False
                    warnings.append(warning_msg)
                    warning_msg = ''
        return warnings

    def check_no_exception(self,stderr_path=None,err_val=None):
        """
            Read standard error file at stderr_path and/or
            interpret err_val to decide if Gromacs ended in error.
            Raises an exception if Gromacs ended in error.
        """
        error_flag = False
        if stderr_path:
            with open(stderr_path,'r') as stderr:
                for line in stderr.readlines():
                    if 'Error in user input' in line or 'Fatal error' in line:
                        error_flag = True
                        error_msg = '\n'
                    if error_flag and '----' not in line:
                        error_msg += line
                    elif error_flag:
                        raise GromacsError(error_msg)
            if err_val == -11:
                raise GromacsError('Gromacs exited with a segmentation fault.')
            elif err_val:
                raise GromacsError('Gromacs exited with code %d, indicating an error,\n'%(err_val)+\
                                   'but I could not identify the error based on the stderr output.\n'+\
                                   'Check file %s'%(stderr_path))
        elif err_val:
            raise GromacsError('Gromacs exited with code %d, indicating an error'%err_val)
        else:
            raise TypeError('check_no_exception needs either err_val or stderr_path')

    def _call(self,command,args,stdin_text=None):
        if len(args) == 1 and type(args[0]) is list:
            raise TypeError('Give arguments as separate python arguments or together as a single space-separated string, but not as a list please.')
        args = regularize_args(args)
        args = command+args
        verbosity = self.verbosity
        stdout_path = '.stdout'
        stderr_path = '.stderr'
        with open(stderr_path,'w') as stderr:
            with open(stdout_path,'w') as stdout:
                if verbosity > 0:
                    print(' '.join(args))
                if not stdin_text:
                    err_val = call(args,stdout=stdout,stderr=stderr)
                else:
                    p = Popen(args,stdin=PIPE,stdout=stdout,stderr=stderr)
                    p.communicate(stdin_text)
                    err_val = p.returncode
        # Check for warnings and possibly ignore+redo
        warnings = self.check_warnings(stderr_path)
        redo = False
        if any(['PME for a system without' in msg for msg in warnings]) \
                and self.ignore_pme_warning:
            print('Ignoring PME for a system wihout charges warning...')
            redo=True
            if '-maxwarn' in args:
                mw_index = args.index('-maxwarn') + 1
                args[mw_index] = int(args[mw_index])+1
            else:
                args += ('-maxwarn','1')
                mw_index = len(args)-1
            print('Increased maxwarn to %s'%(str(args[mw_index])))

        if any(['sum of the two largest charge group' in msg for msg in warnings]) \
                and self.ignore_cg_radii_warning:
            print('Ignoring PME for a system wihout charges warning...')
            redo=True
            if '-maxwarn' in args:
                mw_index = args.index('-maxwarn') + 1
                args[mw_index] = int(args[mw_index])+1
            else:
                args += ('-maxwarn','1')
                mw_index = len(args)-1
            print('Increased maxwarn to %s'%(str(args[mw_index])))

        if any(['I am assuming you are equilibrating' in msg for msg in warnings]) \
                and self.ignore_pr_warning:
            print('Ignoring gen-vel for Parinello-Rahman thermostat  warning...')
            redo=True
            if '-maxwarn' in args:
                mw_index = args.index('-maxwarn') + 1
                args[mw_index] = int(args[mw_index])+1
            else:
                args += ('-maxwarn','1')
                mw_index = len(args)-1
            print('Increased maxwarn to %s'%(str(args[mw_index])))

        if redo:
            with open(stderr_path,'w') as stderr:
                with open(stdout_path,'w') as stdout:
                    if verbosity > 0:
                        print(' '.join(args))
                    err_val = call(args,stdout=stdout,stderr=stderr)

        self.check_no_exception(stderr_path,err_val)

    def nohup_mdrun(self,*args):
        _mdrun = self._mdrun
        self._call(['nohup']+_mdrun[self.key],args)

    def mdrun(self,*args):
        _mdrun = self._mdrun
        self._call(_mdrun[self.key],args)

    def mdrun_mpi(self,np,*args):
        _mdrun_mpi = self._mdrun_mpi
        for key in _mdrun_mpi:
            _mdrun_mpi[key][2] = str(np)
        self._call(_mdrun_mpi[self.key],args)

    def grompp(self,*args):
        _grompp = self._grompp
        self._call(_grompp[self.key],args)

    def genbox(self,*args):
        _genbox = self._genbox
        self._call(_genbox[self.key],args)

    def solvate(self,*args):
        self.genbox(*args)

    def editconf(self,*args):
        _editconf = self._editconf
        self._call(_editconf[self.key],args)

    def genconf(self,*args):
        _genconf = self._genconf
        self._call(_genconf[self.key],args)

    def trjcat(self,*args):
        _trjcat = self._trjcat
        self._call(_trjcat[self.key],args)

    def trjconv(self,*args,**kwargs):
        _trjconv = self._trjconv
        stdin_text = kwargs['stdin_text']
        self._call(_trjconv[self.key],args,stdin_text=stdin_text)

    def energy(self,*args,**kwargs):
        _energy = self._energy
        stdin_text = kwargs['stdin_text']
        self._call(_energy[self.key],args,stdin_text=stdin_text)

    def gyrate(self,*args,**kwargs):
        _gyrate = self._gyrate
        stdin_text = kwargs['stdin_text']
        self._call(_gyrate[self.key],args,stdin_text=stdin_text)

    def wham(self,*args):
        _wham = self._wham
        self._call(_wham[self.key],args)

    def tpbconv(self,*args):
        _tpbconv = self._tpbconv
        self._call(_tpbconv[self.key],args)

    def dump(self,*args):
        _dump = self._dump
        self._call(_dump[self.key],args)

    def dump_key(self,tpr,key):
        items = {}
        self.dump('-s %s'%tpr)
        with open('.stdout') as stdout:
            for line in stdout.readlines():
                if '=' in line:
                    dump_key = line.split('=')[0].strip()
                    dump_val = line.split('=')[1].strip().split()
                    if dump_key == key:
                        return  dump_val
        raise KeyError('Didn\'t find key in tpr dump')

class Gromacs_mdp_base(object):
    """
        Base class for Gromacs mdp.
        - Converts key-value pairs in dictionary into an mdp file.
        - mdp file consists of keyword on left, separated from values
          on right by an equals sign. Multiple mdp values are specified
          by a tuple or list as the value in the python dictionary.
    """
    def __init__(self,params=None):
        self.params = {}
    def __str__(self):
        left_space = max([len(key) for key in self.params])
        text = ''
        for key,value in self.params.items():
            if type(value) in (list,tuple):
                value = [str(i) for i in value]
                value = ' '.join(value)
            text += key.ljust(left_space) + ' = ' + str(value) + '\n'
        return text
    def write(self,path):
        with open(path,'w') as file:
            file.write(self.__str__())
    def add_params(self,params):
        for key,value in params.items():
            self.params[key] = value
