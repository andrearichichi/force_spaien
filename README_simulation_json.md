# simulation.json

Questo documento descrive il formato di `simulation.json` generato dagli script SAPIEN in `force_spaien/scripts/`.

Gli script che lo producono oggi sono:

- `scripts/render_revolute_video.py`
- `scripts/render_prismatic_video.py`
- `scripts/render_screw_video.py`

Il file ha due scopi:

- salvare i metadati della simulazione e dell'oggetto simulato
- salvare la serie temporale dei campioni (`samples`) da cui si possono ricostruire traiettoria, velocita e forze applicate

## Struttura ad alto livello

Lo schema corrente produce un JSON con questa forma:
 
```json
{
  "metadata": {
    "schema_version": 3,
    "...": "..."
  },
  "samples": {
    "...": "..."
  }
}
```

`metadata` contiene contesto, configurazione, fisica e struttura articolata.

`samples` contiene i dati nel tempo. La forma esatta dipende da:

- tipo di movimento/giunto: `revolute`, `prismatic` o `screw`
- modalita: `render` o `apply`

## Schema corrente: metadata

Il blocco `metadata` viene costruito in `scripts/simulation_json.py`.

### `metadata.schema_version`

- `3` nello schema corrente

### `metadata.pipeline`

- `mode`: `render` oppure `apply`

### `metadata.object`

- `model_dir`: path del modello usato
- `joint_type`: `revolute`, `prismatic` oppure `screw`
- `joint`: nome del joint simulato
- `link`: nome del link su cui viene applicata la forza
- `drawer`: indice del cassetto, presente solo per i casi prismatici

### `metadata.output`

- `json_output`: path del JSON scritto
- `video_output`: path del video scritto, presente solo in modalita `render`

### `metadata.timing`

- `fps`: frame rate del video o rate di campionamento desiderato
- `requested_seconds`: durata richiesta via CLI
- `simulated_seconds`: durata fisica realmente simulata
- `sample_interval_s`: intervallo tra un sample e il successivo nel JSON
- `timestep_s`: timestep fisico della scena SAPIEN
- `end_hold_seconds`: solo in `render`, tempo di freeze dell'ultimo frame
- `video_duration_seconds`: solo in `render`, pari a `simulated_seconds + end_hold_seconds`

### `metadata.actuation`

Questo blocco cambia a seconda dello script e della modalita.

Caso `revolute`, `render`:

- `initial_joint_position.rad`
- `initial_joint_position.deg`
- `opening_force.magnitude_n`
- `opening_force.direction_world`
- `closing_force.magnitude_n`
- `closing_force.direction_world`
- `joint_limits_rad`

Caso `revolute`, `apply`:

- `initial_joint_position.rad`
- `initial_joint_position.deg`
- `force.magnitude_n`
- `force.direction_world`
- `joint_limits_rad`

Caso `prismatic`, `render`:

- `force.magnitude_n`
- `force.direction_world`
- `force.generalized_joint_force_n`
- `joint_limits_m`

Caso `prismatic`, `apply`:

- `force.magnitude_n`
- `force.direction_world`
- `force.generalized_joint_force_n`
- `joint_limits_m`

Caso `screw`, `render` o `apply`:

- `motion_type`: sempre `screw`
- `control_mode`: `virtual_screw_constraint_dynamics`
- `virtual_physics`: `true`
- `real_thread_contact`: `false`
- `master_joint`: joint revolute/continuous che controlla `theta`
- `coupled_joint`: joint prismatico aggiornato da `theta`
- `torque.magnitude_nm`
- `torque.axis_world`
- `linear_joint`
- `rotary_joint`
- `linear_limits_m`
- `rotary_limits_rad`
- `screw_dynamics.theta`
- `screw_dynamics.omega`
- `screw_dynamics.translation`
- `screw_dynamics.pitch`
- `screw_dynamics.z0`
- `screw_dynamics.torque`
- `screw_dynamics.damping`
- `screw_dynamics.friction`
- `screw_dynamics.inertia`
- `constraint_equation`: `translation = z0 + pitch * theta / (2*pi)`
- `constraint_error_max`
- `constraint_error_mean`

Nota importante: il caso `screw` e' un vincolo elicoidale virtuale. SAPIEN non simula il contatto geometrico della filettatura; il controller impone direttamente il coupling cinematico tra rotazione e traslazione.

### `metadata.application_point`

- `strategy`: come e' stato scelto il punto di applicazione
- `local_on_link`: coordinate locali del punto sul link

Nel caso `screw`, `application_point` e' usato come ancora visuale per la coppia/marker. Non rappresenta un punto fisico di contatto filettato.

### `metadata.summary`

Questo blocco riassume i numeri principali della simulazione senza dover rileggere tutti i sample.

Campi comuni:

- `physics_step_count`: numero totale di step fisici eseguiti
- `total_sample_count`: numero totale di sample scritti nel JSON
- `sample_series`: mappa per serie di sample, per esempio `force`, `opening_force`, `closing_force`, `no_force`, `pulling_force`, `screw_motion`

Ogni entry di `sample_series` contiene almeno:

- `sample_count`
- `initial_*`
- `final_*`
- `delta_*`
- `max_abs_*`
- `time_of_max_abs_joint_velocity_s`

`initial_*` rappresenta lo stato iniziale reale della simulazione, non semplicemente il primo sample registrato.

Esempio `revolute`:

```json
{
  "summary": {
    "physics_step_count": 960,
    "total_sample_count": 240,
    "sample_series": {
      "opening_force": {
        "sample_count": 120,
        "initial_joint_angle_rad": -1.5,
        "final_joint_angle_rad": -0.82,
        "delta_joint_angle_rad": 0.68,
        "initial_joint_angle_deg": -85.94,
        "final_joint_angle_deg": -46.98,
        "delta_joint_angle_deg": 38.96,
        "max_abs_joint_velocity_rad_s": 0.74,
        "time_of_max_abs_joint_velocity_s": 2.3
      }
    }
  }
}
```

### `metadata.physics`

- `urdf_joint_dynamics`: parametri letti da `mobility.urdf` se presenti
- `uses_separate_static_dynamic_friction`
- `uses_air_friction_model`
- `overrides.link_linear_damping`
- `overrides.link_angular_damping`
- `overrides.joint_drive.stiffness`
- `overrides.joint_drive.damping`
- `overrides.joint_drive.force_limit`

Nota: oggi i flag di `physics` sono costanti fissate dagli script, non misure stimate dalla simulazione.

### `metadata.articulation`

Contiene lo snapshot della struttura articolata usata nella simulazione.

`metadata.articulation.links[]` contiene per ogni link:

- `name`
- `mass`
- `inertia`
- `cmass_local_pose.p`
- `cmass_local_pose.q`
- `linear_damping`
- `angular_damping`
- `disable_gravity`

`metadata.articulation.joints[]` contiene per ogni joint:

- `name`
- `limits_rad` oppure `limits_m`
- `friction`
- `damping`
- `drive_mode`
- `drive_target`
- `drive_velocity_target`
- `force_limit`

## Schema corrente: samples

I `samples` sono la serie temporale vera e propria. Ogni entry contiene sempre un tempo e lo stato del movimento simulato.

Per `revolute` e `prismatic`, ogni entry contiene in genere:

- `time_s`
- `application_point_world`
- `applied_force_world`

Per `screw`, ogni entry contiene invece:

- `time_s`
- `application_point_world`
- `applied_torque_world`
- `applied_axial_force_world`

Poi aggiunge le grandezze del joint in base al caso.

### `revolute` in modalita `render`

La struttura e':

```json
{
  "samples": {
    "opening_force": [
      {
        "time_s": 0.0333,
        "joint_angle_rad": -1.42,
        "joint_angle_deg": -81.4,
        "joint_velocity_rad_s": 0.17,
        "application_point_world": [-0.75, -0.44, 0.04],
        "applied_force_world": [0.0, 0.0, 0.5]
      }
    ],
    "closing_force": [
      {
        "time_s": 0.0333,
        "joint_angle_rad": -1.50,
        "joint_angle_deg": -85.9,
        "joint_velocity_rad_s": -0.11,
        "application_point_world": [-0.75, -0.44, 0.04],
        "applied_force_world": [0.0, 0.0, -0.5]
      }
    ]
  }
}
```

### `revolute` in modalita `apply`

La struttura e':

```json
{
  "samples": {
    "force": [
      {
        "time_s": 0.0042,
        "joint_angle_rad": 0.01,
        "joint_angle_deg": 0.57,
        "joint_velocity_rad_s": 0.21,
        "application_point_world": [0.1, 0.2, 0.3],
        "applied_force_world": [0.0, 0.0, 0.5]
      }
    ]
  }
}
```

In `schema_version = 3`, `revolute/render` e `revolute/apply` usano gli stessi nomi:

- `joint_angle_rad`
- `joint_angle_deg`
- `joint_velocity_rad_s`

### `prismatic` in modalita `render`

La struttura e':

```json
{
  "samples": {
    "no_force": [
      {
        "time_s": 0.0333,
        "joint_position_m": 0.0,
        "joint_velocity_m_s": 0.0,
        "application_point_world": [0.0, 0.0, 0.0],
        "applied_force_world": [0.0, 0.0, 0.0]
      }
    ],
    "pulling_force": [
      {
        "time_s": 0.0333,
        "joint_position_m": 0.01,
        "joint_velocity_m_s": 0.08,
        "application_point_world": [0.0, 0.0, 0.0],
        "applied_force_world": [0.5, 0.0, 0.0]
      }
    ]
  }
}
```

### `prismatic` in modalita `apply`

La struttura e':

```json
{
  "samples": {
    "force": [
      {
        "time_s": 0.0042,
        "joint_position_m": 0.01,
        "joint_velocity_m_s": 0.08,
        "application_point_world": [0.0, 0.0, 0.0],
        "applied_force_world": [0.5, 0.0, 0.0],
        "generalized_force_n": 0.5
      }
    ]
  }
}
```

In `schema_version = 3`, `prismatic/render` e `prismatic/apply` usano gli stessi nomi:

- `joint_position_m`
- `joint_velocity_m_s`

### `screw` in modalita `render` o `apply`

La struttura usa una singola serie:

```json
{
  "samples": {
    "screw_motion": [
      {
        "frame": 0,
        "time": 0.0333,
        "time_s": 0.0333,
        "motion_type": "screw",
        "screw_progress": 0.0,
        "theta_rad": 0.0,
        "theta_deg": 0.0,
        "omega_rad_s": 0.0,
        "translation_m": -0.008,
        "expected_translation_m": -0.008,
        "constraint_error_m": 0.0,
        "pitch": -0.2,
        "z0": -0.008,
        "joint_position_m": -0.008,
        "joint_velocity_m_s": 0.0,
        "joint_angle_rad": 0.0,
        "joint_angle_deg": 0.0,
        "joint_velocity_rad_s": 0.0,
        "friction_torque_nm": 0.0,
        "application_point_world": [0.0, 0.0, 0.0],
        "torque_visual_anchor_world": [0.0, 0.0, 0.0],
        "applied_torque_world": [0.0, -0.01, 0.0],
        "applied_axial_force_world": [0.0, 0.0, 0.0]
      }
    ]
  }
}
```

Campi principali:

- `frame`: indice del frame/sample scritto
- `time` e `time_s`: tempo del sample in secondi
- `theta_rad`: coordinata indipendente del vincolo screw
- `theta_deg`: `theta_rad` convertito in gradi
- `omega_rad_s`: velocita angolare di `theta`
- `translation_m`: posizione prismatic effettivamente applicata
- `expected_translation_m`: posizione attesa dalla formula screw
- `constraint_error_m`: `translation_m - expected_translation_m`
- `pitch`: traslazione per giro completo
- `z0`: posizione prismatic quando `theta = 0`
- `joint_position_m`: alias compatibile con i sample prismatici
- `joint_angle_rad` e `joint_angle_deg`: angolo assoluto del joint revolute
- `joint_velocity_rad_s`: alias compatibile con i sample revolute
- `applied_torque_world`: coppia applicata, espressa in coordinate mondo
- `applied_axial_force_world`: forza assiale opzionale, se configurata

Per lo screw, la relazione verificata in ogni sample e':

```text
translation = z0 + pitch * theta / (2*pi)
```

Il JSON include anche `constraint_error_max` e `constraint_error_mean` in `metadata` e in `metadata.actuation`.

## Screw Joint Simulation

The **screw** motion is implemented as a **virtual helical constraint**: SAPIEN does not simulate real threaded contact, so rotation and translation are coupled directly by the controller.

### Common parameters

| Parameter | Meaning |
| --- | --- |
| `joint_type: "screw"` | Enable screw controller |
| `rotary_joint` | Revolute joint controlling angle `theta` |
| `linear_joint` | Prismatic joint updated from `theta` |
| `link` | Moving part, for example a cap |
| `torque.magnitude_nm` | Applied torque |
| `torque.axis_world` | Torque axis in world coordinates |
| `rotational_inertia_kg_m2` | Virtual rotational inertia |
| `friction_torque_nm` | Constant rotational friction |
| `friction_velocity_scale_rad_s` | Smooths friction around zero velocity |
| `damping_nm_s_rad` | Viscous damping |
| `translation.start / end` | Prismatic motion range |
| `rotation.start / end` | Angular range used for the video/simulation |
| `pitch_m_per_revolution` | Translation per full turn |
| `z0` | Prismatic position at `theta = 0` |
| `application_point` | Torque marker only |

### Configurations

#### `3763`

```yaml
torque.magnitude_nm: 0.01
pitch_m_per_revolution: -0.2
z0: -0.008

translation: -0.008 -> -0.108
rotation: 0 -> 180 deg
```

#### `3990`

```yaml
torque.magnitude_nm: 0.01
pitch_m_per_revolution: -0.32
z0: 0.084

translation: 0.084 -> -0.076
rotation: 0 -> 180 deg
```

## Versioni e migrazione

Ci sono tre famiglie di formato da conoscere.

### 1. Legacy senza `schema_version`

Questi file sono quelli che oggi si vedono gia' salvati in `force_spaien/outputs/*/simulation.json`.

- `metadata` e' piu' piatto
- campi come `fps`, `seconds`, `joint`, `link`, `timestep_s` sono al primo livello di `metadata`
- non hanno `metadata.summary`

### 2. `schema_version = 2`

Questa versione introduceva la struttura annidata di `metadata`, ma non includeva ancora:

- `metadata.summary`
- la normalizzazione completa dei nomi nei sample `render`

In particolare:

- `revolute/render` usava `hinge_angle_*`
- `prismatic/render` usava `drawer_displacement_*`

### 3. `schema_version = 3`

Questa e' la versione corrente documentata qui. Aggiunge:

- `metadata.summary`
- nomi uniformi dei sample tra `render` e `apply`
- estensione `screw` con vincolo elicoidale virtuale, `screw_motion` e metriche `constraint_error_*`

Quindi chi legge i JSON deve controllare se esiste `metadata.schema_version`:

- se assente, trattare il file come legacy
- se vale `2`, applicare la migrazione dei nomi dei sample `render` e considerare assente `metadata.summary`
- se vale `3`, usare lo schema documentato sopra

## Estensioni utili ancora possibili

Le due modifiche principali sono gia' state applicate nello schema corrente:

- `metadata.summary`
- normalizzazione dei nomi dei sample tra `render` e `apply`

Restano comunque alcune estensioni che si possono aggiungere facilmente.

### 1. Stato normalizzato rispetto ai limiti del joint

Da aggiungere nei sample:

- `joint_progress`: valore in `[0, 1]` rispetto ai limiti del joint

Perche' serve:

- permette confronti tra oggetti con range diversi
- aiuta a capire subito se la simulazione e' quasi a fine corsa oppure no

### 2. Indice discreto del sample

Da aggiungere nei sample:

- `sample_index`
- `physics_step_index`

Perche' serve:

- rende la traccia temporale robusta anche se in futuro cambia il `timestep_s`
- facilita debug e allineamento con video/frame

Nota: i sample `screw` hanno gia' `frame`; `sample_index` e `physics_step_index` restano utili come estensione uniforme per tutti i tipi.

### 3. Stato iniziale e finale del punto di applicazione in world

Da aggiungere in `metadata.application_point`:

- `world_t0`
- `world_tfinal`

Perche' serve:

- evita di dover prendere il primo e ultimo sample quando serve solo il punto iniziale o finale
- aiuta a verificare se il punto si e' mosso come atteso

### 4. Flag di saturazione ai limiti

Da aggiungere in `metadata.summary` oppure nei sample:

- `hit_lower_limit`
- `hit_upper_limit`
- `distance_to_limit`

Perche' serve:

- molte simulazioni diventano piu' facili da interpretare se si sa subito se il moto si e' fermato per limite meccanico

## Evoluzione dello schema

Le sezioni sotto spiegano dove inserire altri campi se lo schema continua a evolvere.

## Come inserire nuovi campi

I punti principali del codice sono questi:

- `scripts/simulation_json.py`
  Qui si definisce la struttura di `metadata`.

- `scripts/render_revolute_video.py`
  Qui si costruiscono i sample per i casi revolute e si scrive il file JSON finale.

- `scripts/render_prismatic_video.py`
  Qui si costruiscono i sample per i casi prismatici e si scrive il file JSON finale.

- `scripts/render_screw_video.py`
  Qui si costruiscono i sample per i casi screw, si impone il vincolo elicoidale virtuale e si scrive il file JSON finale.

In pratica:

1. Per aggiungere campi globali o strutturali, modificare `build_metadata()` in `scripts/simulation_json.py`.
2. Per aggiungere campi per-sample, modificare `sample_to_dict()` e i blocchi `samples.append(...)` negli script di render/apply.
3. Per aggiungere metriche riassuntive, aggiornare `build_summary()` in `scripts/simulation_json.py` oppure preparare nuovi dati prima del `build_metadata()`.
4. Se il formato cambia in modo incompatibile, incrementare `SCHEMA_VERSION`.
