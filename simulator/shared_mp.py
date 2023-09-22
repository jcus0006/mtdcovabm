agents_static = None

def init_pool_processes(the_agents_static):

    '''Initialize each process with a global agents_static class instance including read-only in multiprocessing array format
    '''
    global agents_static
    agents_static = the_agents_static