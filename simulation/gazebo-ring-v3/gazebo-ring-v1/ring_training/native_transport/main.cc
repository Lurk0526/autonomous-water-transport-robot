#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/transport/Node.hh>

namespace
{
std::mutex output_mutex;
constexpr std::array<const char *, 4> wheel_links{
    "rear_left_wheel", "rear_right_wheel", "front_left_wheel", "front_right_wheel"};

template<typename Value>
void Write(Value value)
{
  // Harmonic's Linux and macOS targets use little-endian scalars.
  std::cout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::uint64_t Stamp(const gz::msgs::Header &header)
{
  return static_cast<std::uint64_t>(header.stamp().sec()) * 1'000'000'000 +
         header.stamp().nsec();
}

void WriteImage(const gz::msgs::Image &image, char kind)
{
  std::uint32_t encoding = 0;
  switch (image.pixel_format_type())
  {
    case gz::msgs::RGB_INT8: encoding = 1; break;
    case gz::msgs::BGR_INT8: encoding = 2; break;
    case gz::msgs::R_FLOAT32: encoding = 3; break;
    default: break;
  }

  const auto &data = image.data();
  std::lock_guard<std::mutex> lock(output_mutex);
  Write(kind);
  Write(static_cast<std::uint32_t>(24 + data.size()));
  Write(Stamp(image.header()));
  Write(image.width());
  Write(image.height());
  Write(image.step());
  Write(encoding);
  std::cout.write(data.data(), data.size()).flush();
}

void OnRgb(const gz::msgs::Image &image)
{
  WriteImage(image, 'R');
}

void OnDepth(const gz::msgs::Image &image)
{
  WriteImage(image, 'D');
}

void OnPose(const gz::msgs::Pose_V &poses)
{
  const gz::msgs::Pose *model = nullptr;
  std::array<const gz::msgs::Pose *, wheel_links.size()> wheels{};
  for (const auto &pose : poses.pose())
  {
    const auto &name = pose.name();
    if (name == "cart")
      model = &pose;
    for (std::size_t index = 0; index < wheels.size(); ++index)
    {
      if (name == std::string("cart::") + wheel_links[index])
        wheels[index] = &pose;
    }
  }
  if (!model)
    return;
  for (const auto *wheel : wheels)
  {
    if (!wheel)
      return;
  }

  auto transform = [](const gz::msgs::Pose &pose) {
    const auto &p = pose.position();
    const auto &q = pose.orientation();
    return gz::math::Pose3d(
        gz::math::Vector3d(p.x(), p.y(), p.z()),
        gz::math::Quaterniond(q.w(), q.x(), q.y(), q.z()));
  };
  const auto world_model = transform(*model);
  std::lock_guard<std::mutex> lock(output_mutex);
  Write('P');
  Write(std::uint32_t{232});
  Write(Stamp(model->header()));
  Write(world_model.Pos().X());
  Write(world_model.Pos().Y());
  Write(world_model.Pos().Z());
  Write(world_model.Rot().Yaw());
  for (const auto *wheel : wheels)
  {
    const auto model_wheel = transform(*wheel);
    const auto center = world_model.Pos() + world_model.Rot().RotateVector(model_wheel.Pos());
    const auto axle = (world_model.Rot() * model_wheel.Rot()).RotateVector(gz::math::Vector3d::UnitY);
    Write(center.X());
    Write(center.Y());
    Write(center.Z());
    Write(axle.X());
    Write(axle.Y());
    Write(axle.Z());
  }
  std::cout.flush();
}

void OnContacts(const gz::msgs::Contacts &contacts)
{
  std::uint8_t grounded = 0;
  for (const auto &contact : contacts.contact())
  {
    const auto &first = contact.collision1().name();
    const auto &second = contact.collision2().name();
    const auto is_road = [](const std::string &name) {
      return name.find("ground::") == 0 || name.find("brick_rough_section::") == 0;
    };
    const std::string *wheel_collision = is_road(first) ? &second : (is_road(second) ? &first : nullptr);
    if (wheel_collision)
    {
      for (std::size_t index = 0; index < wheel_links.size(); ++index)
      {
        if (wheel_collision->find(wheel_links[index]) == std::string::npos)
          continue;
        for (const auto &normal : contact.normal())
        {
          if (std::abs(normal.z()) > 0.5)
            grounded |= static_cast<std::uint8_t>(1u << index);
        }
      }
    }
    if (first.find("cone") == std::string::npos &&
        second.find("cone") == std::string::npos &&
        first.find("barrier") == std::string::npos &&
        second.find("barrier") == std::string::npos)
      continue;

    const auto names = first + " / " + second;
    std::lock_guard<std::mutex> lock(output_mutex);
    Write('C');
    Write(static_cast<std::uint32_t>(8 + names.size()));
    Write(Stamp(contacts.header()));
    std::cout.write(names.data(), names.size()).flush();
  }
  if (grounded != 0)
  {
    std::lock_guard<std::mutex> lock(output_mutex);
    Write('G');
    Write(std::uint32_t{9});
    Write(Stamp(contacts.header()));
    Write(grounded);
    std::cout.flush();
  }
}
}  // namespace

int main()
{
  gz::transport::Node node;
  auto publisher = node.Advertise<gz::msgs::Twist>("/model/cart/cmd_vel");
  const bool subscribed = node.Subscribe("/model/cart/pose", OnPose) &&
                          node.Subscribe("/cart/camera/image", OnRgb) &&
                          node.Subscribe("/cart/camera/depth_image", OnDepth) &&
                          node.Subscribe("/cart/contacts", OnContacts) &&
                          node.Subscribe("/cart/contacts/front_left_wheel", OnContacts) &&
                          node.Subscribe("/cart/contacts/front_right_wheel", OnContacts) &&
                          node.Subscribe("/cart/contacts/rear_left_wheel", OnContacts) &&
                          node.Subscribe("/cart/contacts/rear_right_wheel", OnContacts);
  if (!publisher || !subscribed)
  {
    std::cerr << "Couldn't initialize Gazebo Transport topics\n";
    return 1;
  }

  std::string command;
  while (std::getline(std::cin, command))
  {
    std::istringstream values(command);
    double speed, yaw_rate;
    if (!(values >> speed >> yaw_rate) ||
        !std::isfinite(speed) || !std::isfinite(yaw_rate))
    {
      std::cerr << "Invalid speed/yaw-rate command\n";
      return 1;
    }

    gz::msgs::Twist movement;
    movement.mutable_linear()->set_x(speed);
    movement.mutable_angular()->set_z(yaw_rate);
    if (!publisher.Publish(movement))
    {
      std::cerr << "Gazebo Transport publish failed\n";
      return 1;
    }
  }
}
