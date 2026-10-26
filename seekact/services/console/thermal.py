"""Display-only temperature classification matching existing G1 alarm conventions.

Never dispatches motions. Native ankle hysteresis is retained across stale samples;
stale data itself is always shown as unknown, not as a current alarm or healthy value.
"""
from .adapters import finite


class ThermalDisplay:
    def __init__(self):
        self.rule = None
        self.latched = set()

    def apply(self, telemetry, rule):
        if rule not in ('native90', 'strict91'):
            raise ValueError('unknown thermal display rule')
        if rule != self.rule:
            self.latched.clear()
            self.rule = rule
        counts = dict(normal=0, warning=0, critical=0, unknown=0)
        for joint in telemetry['joints']:
            values = [joint.get('case_c'), joint.get('winding_c')]
            valid = bool(joint.get('fresh')) and all(finite(v) for v in values)
            maximum = max(values) if all(finite(v) for v in values) else None
            index = joint['id']
            ankle = 'ankle' in joint['name']
            if valid and rule == 'native90' and ankle:
                if maximum >= 90:
                    self.latched.add(index)
                elif maximum < 85:
                    self.latched.discard(index)
            retained = rule == 'native90' and index in self.latched
            if not valid:
                level, reason = 'unknown', 'stale'
            elif joint.get('motorstate'):
                level, reason = 'critical', 'motor_fault'
            elif (maximum >= 90 if rule == 'native90' else maximum > 91):
                level, reason = 'critical', 'temperature'
            elif retained:
                level, reason = 'critical', 'alarm_latched'
            elif maximum >= 85:
                level, reason = 'warning', 'temperature'
            else:
                level, reason = 'normal', None
            joint.update(max_c=maximum, level=level, reason=reason, alarm_latched=retained)
            counts[level] += 1
        telemetry.update(thermal_rule=rule, counts=counts,
                         thresholds={'warning_c': 85, 'critical_c': 90 if rule == 'native90' else 91,
                                     'critical_inclusive': rule == 'native90',
                                     'ankle_clear_below_c': 85 if rule == 'native90' else None})
        return telemetry
