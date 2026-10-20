#!/usr/bin/env python3
"""Provision a plan, or snapshot existing slices, saving resources beside run data."""
import argparse
import ipaddress
import json
import re
from pathlib import Path
from resources import capture, digest, utc, write_inventory


def require(condition, message):
    if not condition: raise ValueError(message)


def validate(plan):
    nodes = plan['nodes']
    require(nodes and len({n['name'] for n in nodes}) == len(nodes), 'Node names must be unique')
    names = {n['name'] for n in nodes}
    hosts = [n['host'] for n in nodes if n.get('host')]
    require(len(hosts) == len(set(hosts)), 'Separate replicas/stages must use distinct hosts')
    for n in nodes:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', n['name']) and n['site'], 'Invalid provisioning plan')
        require(n.get('instance_type') or all(int(n[k]) > 0 for k in ['cores', 'ram', 'disk']), 'Invalid provisioning plan')
        require(len({c['name'] for c in n.get('components', [])}) == len(n.get('components', [])), 'Invalid provisioning plan')
    network_names = [n['name'] for n in plan.get('networks', [])]
    require(len(set(network_names)) == len(network_names), 'Network names must be unique')
    for net in plan.get('networks', []):
        subnet = ipaddress.ip_network(net['subnet'])
        addresses = []
        require(len(net['members']) >= 2, 'Invalid provisioning plan')
        require(len({m['node'] for m in net['members']}) == len(net['members']), 'Invalid provisioning plan')
        for m in net['members']:
            require(m['node'] in names, 'Invalid provisioning plan')
            address = ipaddress.ip_address(m['address'])
            require(address in subnet and address not in [subnet.network_address, subnet.broadcast_address], 'Invalid provisioning plan')
            addresses.append(address)
        require(len(set(addresses)) == len(addresses), 'Duplicate network address')
        sites={n['site'] for n in nodes if n['name'] in {m['node'] for m in net['members']}}
        kind=net.get('type')
        require(kind in [None,'L2Bridge','L2STS'], 'Only Basic-NIC L2Bridge and L2STS networks are supported')
        require(kind!='L2Bridge' or len(sites)==1, 'L2Bridge needs one site')
        require(kind!='L2STS' or len(sites)==2, 'L2STS needs exactly two sites')
        for m in net['members']:
            spec=next(n for n in nodes if n['name']==m['node'])
            require(net['name'] not in {c['name'] for c in spec.get('components',[])}, 'NIC/component name collision')
    return plan


def build(fab, plan, name):
    sl = fab.new_slice(name=name)
    nodes, ports = {}, {}
    for spec in plan['nodes']:
        fields = {k: spec[k] for k in ['name', 'site', 'host', 'image', 'instance_type', 'cores', 'ram', 'disk'] if k in spec}
        node = nodes[spec['name']] = sl.add_node(**fields)
        for component in spec.get('components', []):
            node.add_component(**component)
    for net in plan.get('networks', []):
        interfaces = []
        for member in net['members']:
            iface = nodes[member['node']].add_component(model='NIC_Basic', name=net['name']).get_interfaces()[0]
            ports[(net['name'], member['node'])] = iface.get_name()
            interfaces.append(iface)
        sl.add_l2network(name=net['name'], type=net.get('type'), interfaces=interfaces)
    return sl, ports


def provision(fab, args, plan):
    out = Path(args.out)
    state_path = out / 'provision-state.json'
    if state_path.exists():
        raise ValueError('This output directory already has provisioning state; use snapshot to inspect it')
    if (out/'resources.json').exists():
        raise ValueError('Resource inventory already exists; use a new experiment directory')
    if any(s.get_name() == args.name for s in fab.get_slices()):
        raise ValueError('Slice name already exists; use snapshot instead of modifying it')
    sl, ports = build(fab, plan, args.name)
    state = {'started_utc': utc(), 'name': args.name, 'plan': plan,
             'plan_sha256': digest(args.plan), 'provisioner_sha256': digest(__file__),
             'lease_hours': args.lease_hours, 'keep_reserved': True, 'status': 'submitting'}
    def save():
        state_path.write_text(json.dumps(state, indent=2, default=str) + '\n')
    with state_path.open('x') as f:f.write(json.dumps(state,indent=2)+'\n')
    try:
        # Persist the ID before any long wait; never delete a partial allocation.
        returned = sl.submit(wait=False, lease_in_hours=args.lease_hours)
        state.update(slice_id=sl.get_slice_id() or returned, status='submitted')
        save()
        sl.wait(timeout=args.timeout, progress=False)
        sl.update()
        if sl.get_state() != 'StableOK':
            raise RuntimeError('Slice did not reach StableOK')
        sl.wait_ssh(timeout=args.timeout, progress=False)
        sl.post_boot_config()
        for net in plan.get('networks', []):
            for member in net['members']:
                iface = sl.get_node(name=member['node']).get_interface(ports[(net['name'], member['node'])])
                iface.ip_link_up()
                iface.ip_addr_add(addr=ipaddress.ip_address(member['address']), subnet=ipaddress.ip_network(net['subnet']))
        state.update(status='vm_ready', finished_utc=utc())
    except Exception as error:
        state.update(status='incomplete', error_type=type(error).__name__, finished_utc=utc())
        raise
    finally:
        save()
        # Still save FABlib metadata when provisioning or SSH fails.
        try:
            data = capture([sl], guest=state['status'] == 'vm_ready', host_info=args.host_info)
        except Exception as error:
            data = {'source_kind':'partial_live_capture','captured_utc':utc(),
                    'capture_error':type(error).__name__,'slice_id':state.get('slice_id')}
        data['provisioning'] = state
        write_inventory(out, data)
    return [sl]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['provision', 'snapshot'])
    parser.add_argument('--out', required=True, help='Experiment data directory')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--name', help='New unique slice name')
    parser.add_argument('--slice-id', action='append', default=[])
    parser.add_argument('--fabric-rc', type=Path)
    parser.add_argument('--lease-hours', type=int, default=24)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--no-ssh', action='store_true', help='Snapshot FABlib metadata only')
    parser.add_argument('--host-info', action='store_true', help='Also attempt FABlib CPU/NUMA host information')
    parser.add_argument('--dry-run', action='store_true', help='Validate plan locally without login or allocations')
    args = parser.parse_args()
    if args.action == 'provision' and (not args.plan or not args.name):
        parser.error('provision requires --plan and --name')
    if args.action == 'snapshot' and not args.slice_id:
        parser.error('snapshot requires at least one --slice-id')
    if args.lease_hours <= 0 or args.timeout <= 0:
        parser.error('lease hours and timeout must be positive')
    plan = validate(json.loads(args.plan.read_text())) if args.plan else None
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'name': args.name, 'plan': plan}, indent=2))
        return
    from fabrictestbed_extensions.fablib.fablib import FablibManager
    fab = FablibManager(fabric_rc=str(args.fabric_rc) if args.fabric_rc else None,
                        auto_token_refresh=False, log_level='ERROR')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.action == 'provision':
        provision(fab, args, plan)
    else:
        if (Path(args.out) / 'resources.json').exists():
            parser.error('Resource inventory exists; choose a new snapshot directory to preserve history')
        slices = [fab.get_slice(slice_id=sid) for sid in args.slice_id]
        write_inventory(args.out, capture(slices, guest=not args.no_ssh, host_info=args.host_info))
    print(str(Path(args.out) / 'resources.txt'))


if __name__ == '__main__':
    main()
