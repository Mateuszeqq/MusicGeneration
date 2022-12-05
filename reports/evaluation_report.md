# Discriminator (critic) and generator losses
![discriminator_losses](figures/discriminator_losses.png)
![generator_losses](figures/generator_losses.png)
# Metrics

- **Empty Beat Rate**<br>
The empty-beat rate is defined as the ratio of the number of empty beats (where no note is played) to the total number of beats.<br>
![Empty Measure Rate](figures/empty_beat_rate.png)<br>

- **Empty Beat Rate**<br>
Ratio of empty measures.<br>
![empty_measure_rate](figures/empty_measure_rate.png)<br>

- **Groove Consistency**<br>
Ratio of empty measures.<br>
$groove\_consistency = 1 - \frac{1}{T - 1} \sum_{i = 1}^{T - 1}{d(G_i, G_{i + 1})}$ <br>
Here, $T$ is the number of measures $G_i$,  is the binary onset vector of the -th measure (a one at position that has an onset, otherwise a zero), and $d(G,G')$  is the hamming distance between two vectors $G$ and $G'$. Note that this metric only works for songs with a constant time signature.<br>
![groove_consistency](figures/groove_consistency.png)<br>

- **N Pitches Used**<br>
Unique pitches used.<br>
![n_pitches_used](figures/n_pitches_used.png)<br>

- **Pitch Class Entropy**<br>
Entropy of the normalized note pitch class histogram.
The pitch class entropy is defined as the Shannon entropy of the normalized note pitch class histogram.<br>
$pitch\_class\_entropy = -\sum_{i = 0}^{11}{
    P(pitch\_class=i) \times \log_2 P(pitch\_class=i)}$<br>
![pitch_class_entropy](figures/pitch_class_entropy.png)<br>

- **Pitch Entropy**<br>
Entropy of the normalized pitch histogram.<br>
![pitch_entropy](figures/pitch_entropy.png)<br>

- **Pitch Range**<br>
Pitch range.<br>
![pitch_range](figures/pitch_range.png)<br>

- **Polyphony**<br>
Average number of pitches being played concurrently.
The polyphony is defined as the average number of pitches being played at the same time, evaluated only at time steps where at least one pitch is on.<br>
$polyphony = \frac{
    \#(pitches\_when\_at\_least\_one\_pitch\_is\_on)
}{
    \#(time\_steps\_where\_at\_least\_one\_pitch\_is\_on)
}$ <br>
![polyphony](figures/polyphony.png)<br>

- **Polyphony Rate**<br>
Ratio of time steps where multiple pitches are on.
The polyphony rate is defined as the ratio of the number of time steps where multiple pitches are on to the total number of time steps.<br>
$polyphony\_rate = \frac{
    \#(time\_steps\_where\_multiple\_pitches\_are\_on)
}{
    \#(time\_steps)
}$<br>
![polyphony_rate](figures/polyphony_rate.png)<br>

- **Scale Consistency**<br>
The largest pitch-in-scale rate.
The scale consistency is defined as the largest pitch-in-scale rate over all major and minor scales.<br>
$scale\_consistency = \max_{root, mode}{
    pitch\_in\_scale\_rate(root, mode)}$<br>
![scale_consistency](figures/scale_consistency.png)<br>